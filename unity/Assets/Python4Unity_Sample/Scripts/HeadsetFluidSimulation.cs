using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Layered 2.5D fluid on a 2D texture atlas. Plume particles sample the
/// velocity. The burn-map compute shader is not modified; this only reads it.
/// </summary>
public class HeadsetFluidSimulation : MonoBehaviour
{
    const int MaxResolution = 96;
    const int MaxLayers = 8;
    const int MaxPressureIterations = 20;
    const int MaxViscosityIterations = 8;
    const int MaxParticleCapacity = 4096;
    const int MaxSources = 32;
    const int MaxSpawnPerFrame = 64;

    [Header("Grid")]
    [Tooltip("Horizontal cells on each side of the fluid atlas. Quest 3: 32 to 64. Values above 96 are clamped.")]
    [SerializeField] int gridResolution = 64;

    [Tooltip("Vertical layers. The volume is stored as a 2D atlas (height = resolution * layers) because Quest GLES/Vulkan compute should stay on RWTexture2D. Values above 8 are clamped.")]
    [SerializeField] int layerCount = 4;

    [Header("Fluid")]
    [Tooltip("Kinematic viscosity in m^2/s. Real air (~1.5e-5) is invisible on a cell a few meters wide. 12 m^2/s is a stable visual default.")]
    [SerializeField] float viscosity = 12f;

    [Tooltip("Buoyancy scale. Vertical velocity gains buoyancyCoefficient * (temperature - ambientTemperature) * dt. Temperature is normalized heat from the burn map, not Kelvin.")]
    [SerializeField] float buoyancyCoefficient = 4f;

    [Tooltip("Rest temperature of the field. 0 matches the burn map, whose red channel is intensity in 0..1 rather than Kelvin.")]
    [SerializeField] float ambientTemperature = 0f;

    [Tooltip("Exponential decay rate in 1/s applied to velocity and to the temperature excess each step.")]
    [SerializeField] float dissipation = 0.55f;

    [Tooltip("Jacobi iterations for the pressure projection that removes divergence. 8 is the Quest-sized default. Values above 20 are clamped.")]
    [SerializeField] int iterationCount = 8;

    [Tooltip("Jacobi iterations for viscosity. 2 is enough at the default viscosity. 0 skips diffusion. Values above 8 are clamped.")]
    [SerializeField] int viscosityIterations = 2;

    [Tooltip("Fluid integration step in seconds. 0 uses the frame delta clamped to 1/30 s. This is the GPU fluid step, not the 20 s fire-spread DT. Explicit values are clamped to 0.1 s.")]
    [SerializeField] float timeStep = 0f;

    [Tooltip("How strongly packet wind is added to the velocity field, in 1/s. Velocity gains wind * windCoupling * dt.")]
    [SerializeField] float windCoupling = 0.5f;

    [Tooltip("Heat added per second from a fully intense burn-map cell into the lowest layer. Upper layers receive less.")]
    [SerializeField] float heatSourceStrength = 1.2f;

    [Tooltip("Speed clamp in m/s applied after the pressure projection.")]
    [SerializeField] float maxSpeed = 30f;

    [Tooltip("Turn the fluid off to keep the original CPU plume particle systems.")]
    [SerializeField] bool simulate = true;

    [Header("Particles")]
    [Tooltip("GPU plume particles that sample the velocity field. Clamped to 4096.")]
    [SerializeField] int maxParticles = 4096;

    [Tooltip("Particles spawned per frame while plume sources exist. Clamped to 64.")]
    [SerializeField] int spawnsPerFrame = 10;

    [Tooltip("Particle lifetime in seconds.")]
    [SerializeField] float particleLifetime = 6f;

    [Tooltip("Base billboard size in meters. Heat and age scale it on the GPU.")]
    [SerializeField] float particleSize = 2.5f;

    [Tooltip("Packet total_heat, in kW, that maps to heat of 1. Matches the previous plume renderer's scale.")]
    [SerializeField] float referenceHeat = 1000000f;

    [Tooltip("Fluid compute shader. Leave empty to load HeadsetFluid from Resources.")]
    [SerializeField] ComputeShader fluidCompute;

    [Tooltip("Billboard shader. Leave empty to load Ember/PlumeFluidParticles from Resources.")]
    [SerializeField] Shader particleShader;

    struct PlumeSourceGPU
    {
        public Vector4 originHeat;
        public Vector4 radiusMisc;
    }

    BurnMapManager _burn;
    RenderTexture _stateA;
    RenderTexture _stateB;
    RenderTexture _stateRead;
    RenderTexture _stateWrite;
    RenderTexture _pressureA;
    RenderTexture _pressureB;
    RenderTexture _pressureRead;
    RenderTexture _pressureWrite;
    RenderTexture _divergence;
    RenderTexture _posAge;
    RenderTexture _aux;
    Texture2D _dummyBurn;
    ComputeBuffer _sources;
    Material _particleMaterial;
    PlumeSourceGPU[] _sourceData = new PlumeSourceGPU[MaxSources];

    int _res;
    int _layers;
    int _particleCapacity;
    int _sourceCount;
    int _kInit, _kAdvect, _kForces, _kDiffuse, _kDivergence, _kPressure, _kProject, _kClear, _kSpawn, _kAdvectParticles;
    bool _ready;
    bool _failed;
    bool _loggedClamp;
    Vector3 _ambientWind;
    RenderTextureFormat _stateFormat;

    public bool WillDrivePlumes
    {
        get { return simulate && enabled && !_failed && SystemInfo.supportsComputeShaders; }
    }

    public bool IsDrivingPlumes
    {
        get { return WillDrivePlumes && _ready; }
    }

    void OnEnable()
    {
        _burn = GetComponent<BurnMapManager>();
        if (_burn == null)
            _burn = FindObjectOfType<BurnMapManager>();
    }

    void LateUpdate()
    {
        if (!WillDrivePlumes)
            return;
        if (_burn == null || _burn.burnMap == null)
            return;
        if (!EnsureResources())
            return;
        Step(FluidDeltaTime());
    }

    void OnRenderObject()
    {
        if (!IsDrivingPlumes || _particleMaterial == null)
            return;
        Camera cam = Camera.current;
        if (cam == null || cam.cameraType == CameraType.Preview)
            return;

        _particleMaterial.SetTexture("_ParticlePosAge", _posAge);
        _particleMaterial.SetTexture("_ParticleAux", _aux);
        if (_particleMaterial.SetPass(0))
            Graphics.DrawProceduralNow(MeshTopology.Triangles, _particleCapacity * 6, 1);
    }

    void OnDestroy()
    {
        ReleaseResources();
    }

    public void SetPlumeSources(List<PlumeData> plumes)
    {
        _sourceCount = 0;
        if (plumes == null || plumes.Count == 0)
            return;

        var ranked = new List<PlumeData>(plumes.Count);
        for (int i = 0; i < plumes.Count; i++)
        {
            PlumeData plume = plumes[i];
            if (plume == null || plume.origin_position == null || plume.origin_position.Length != 3)
                continue;
            if (plume.total_heat <= 0f)
                continue;
            ranked.Add(plume);
        }
        ranked.Sort((a, b) => b.total_heat.CompareTo(a.total_heat));

        int count = Mathf.Min(ranked.Count, MaxSources);
        Vector3 windSum = Vector3.zero;
        float heatScale = Mathf.Max(referenceHeat, 1f);
        for (int i = 0; i < count; i++)
        {
            PlumeData plume = ranked[i];
            Vector3 origin = new Vector3(plume.origin_position[0], plume.origin_position[2], plume.origin_position[1]);
            windSum += new Vector3(plume.plume_u, plume.plume_w, plume.plume_v);
            float heat = Mathf.Clamp01(plume.total_heat / heatScale);
            float radius = 2f;
            if (plume.ell > 0f)
                radius = Mathf.Clamp(plume.ell * 0.05f, 1.5f, 10f);
            _sourceData[i].originHeat = new Vector4(origin.x, origin.y, origin.z, heat);
            _sourceData[i].radiusMisc = new Vector4(radius, Mathf.Max(plume.h_sub, 0f), Mathf.Max(plume.k, 0f), 1f);
        }

        _sourceCount = count;
        if (count > 0)
            _ambientWind = windSum / count;
        UploadSources();
    }

    float FluidDeltaTime()
    {
        if (timeStep > 0f)
            return Mathf.Clamp(timeStep, 0.0001f, 0.1f);
        return Mathf.Min(Time.deltaTime, 1f / 30f);
    }

    bool EnsureResources()
    {
        int res = Mathf.Clamp(gridResolution, 8, MaxResolution);
        int layers = Mathf.Clamp(layerCount, 1, MaxLayers);
        int particles = Mathf.Clamp(maxParticles, 64, MaxParticleCapacity);
        if (!_loggedClamp && (res != gridResolution || layers != layerCount || particles != maxParticles))
        {
            Debug.LogWarning($"Headset fluid clamped grid {gridResolution}x{layerCount} and {maxParticles} particles to {res}x{layers} and {particles} for Quest limits.");
            _loggedClamp = true;
        }

        if (_ready && res == _res && layers == _layers && particles == _particleCapacity && fluidCompute != null)
            return true;

        ReleaseResources();
        _failed = false;

        if (fluidCompute == null)
            fluidCompute = Resources.Load<ComputeShader>("HeadsetFluid");
        if (particleShader == null)
            particleShader = Resources.Load<Shader>("PlumeFluidParticles");
        if (particleShader == null)
            particleShader = Shader.Find("Ember/PlumeFluidParticles");
        if (fluidCompute == null || particleShader == null)
        {
            _failed = true;
            Debug.LogError("Headset fluid shader is missing. CPU plume particles stay in use.");
            return false;
        }

        try
        {
            _res = res;
            _layers = layers;
            _particleCapacity = particles;
            _stateFormat = SystemInfo.SupportsRandomWriteOnRenderTextureFormat(RenderTextureFormat.ARGBHalf)
                ? RenderTextureFormat.ARGBHalf
                : RenderTextureFormat.ARGBFloat;
            if (!SystemInfo.SupportsRandomWriteOnRenderTextureFormat(_stateFormat)
                || !SystemInfo.SupportsRandomWriteOnRenderTextureFormat(RenderTextureFormat.ARGBFloat))
            {
                throw new InvalidOperationException("This GPU cannot random-write the fluid textures.");
            }

            int atlasHeight = _res * _layers;
            _stateA = CreateGrid(_res, atlasHeight, _stateFormat);
            _stateB = CreateGrid(_res, atlasHeight, _stateFormat);
            _pressureA = CreateGrid(_res, atlasHeight, _stateFormat);
            _pressureB = CreateGrid(_res, atlasHeight, _stateFormat);
            _divergence = CreateGrid(_res, atlasHeight, _stateFormat);
            _posAge = CreateGrid(_particleCapacity, 1, RenderTextureFormat.ARGBFloat);
            _aux = CreateGrid(_particleCapacity, 1, RenderTextureFormat.ARGBFloat);
            _stateRead = _stateA;
            _stateWrite = _stateB;
            _pressureRead = _pressureA;
            _pressureWrite = _pressureB;

            _dummyBurn = new Texture2D(1, 1, TextureFormat.RGBA32, false, true);
            _dummyBurn.SetPixel(0, 0, Color.clear);
            _dummyBurn.Apply(false, true);

            _sources = new ComputeBuffer(MaxSources, sizeof(float) * 8);
            CacheKernels();
            BindGlobals(0.016f);
            ClearGrid(_stateA, _pressureA);
            ClearGrid(_stateB, _pressureB);
            fluidCompute.SetTexture(_kClear, "ParticlePosAge", _posAge);
            fluidCompute.SetTexture(_kClear, "ParticleAux", _aux);
            fluidCompute.Dispatch(_kClear, (_particleCapacity + 63) / 64, 1, 1);

            _particleMaterial = new Material(particleShader);
            _particleMaterial.enableInstancing = true;
            _particleMaterial.hideFlags = HideFlags.HideAndDontSave;

            UploadSources();
            _ready = true;
            Debug.Log($"Headset fluid ready: {_res}x{_res}x{_layers} atlas, format {_stateFormat}, {_particleCapacity} particles.");
            return true;
        }
        catch (Exception exception)
        {
            _failed = true;
            _ready = false;
            Debug.LogError($"Headset fluid failed to allocate and will leave CPU plumes in use: {exception.Message}");
            ReleaseResources();
            return false;
        }
    }

    void CacheKernels()
    {
        _kInit = fluidCompute.FindKernel("InitGrid");
        _kAdvect = fluidCompute.FindKernel("Advect");
        _kForces = fluidCompute.FindKernel("AddForces");
        _kDiffuse = fluidCompute.FindKernel("DiffuseVelocity");
        _kDivergence = fluidCompute.FindKernel("Divergence");
        _kPressure = fluidCompute.FindKernel("PressureJacobi");
        _kProject = fluidCompute.FindKernel("SubtractGradient");
        _kClear = fluidCompute.FindKernel("ClearParticles");
        _kSpawn = fluidCompute.FindKernel("SpawnParticles");
        _kAdvectParticles = fluidCompute.FindKernel("AdvectParticles");
    }

    void Step(float dt)
    {
        if (dt <= 0f)
            return;

        BindGlobals(dt);
        Texture burn = _burn != null && _burn.burnMap != null ? (Texture)_burn.burnMap : _dummyBurn;

        fluidCompute.SetTexture(_kAdvect, "StateRead", _stateRead);
        fluidCompute.SetTexture(_kAdvect, "StateWrite", _stateWrite);
        DispatchGrid(_kAdvect);
        SwapState();

        fluidCompute.SetTexture(_kForces, "StateRead", _stateRead);
        fluidCompute.SetTexture(_kForces, "StateWrite", _stateWrite);
        fluidCompute.SetTexture(_kForces, "BurnMapTex", burn);
        DispatchGrid(_kForces);
        SwapState();

        int viscSteps = Mathf.Clamp(viscosityIterations, 0, MaxViscosityIterations);
        if (viscSteps > 0 && viscosity > 0f)
        {
            for (int i = 0; i < viscSteps; i++)
            {
                fluidCompute.SetTexture(_kDiffuse, "StateRead", _stateRead);
                fluidCompute.SetTexture(_kDiffuse, "StateWrite", _stateWrite);
                DispatchGrid(_kDiffuse);
                SwapState();
            }
        }

        fluidCompute.SetTexture(_kDivergence, "StateRead", _stateRead);
        fluidCompute.SetTexture(_kDivergence, "DivergenceMap", _divergence);
        DispatchGrid(_kDivergence);

        int pressureSteps = Mathf.Clamp(iterationCount, 1, MaxPressureIterations);
        for (int i = 0; i < pressureSteps; i++)
        {
            fluidCompute.SetTexture(_kPressure, "PressureRead", _pressureRead);
            fluidCompute.SetTexture(_kPressure, "PressureWrite", _pressureWrite);
            fluidCompute.SetTexture(_kPressure, "DivergenceMap", _divergence);
            DispatchGrid(_kPressure);
            SwapPressure();
        }

        fluidCompute.SetTexture(_kProject, "StateRead", _stateRead);
        fluidCompute.SetTexture(_kProject, "StateWrite", _stateWrite);
        fluidCompute.SetTexture(_kProject, "PressureRead", _pressureRead);
        DispatchGrid(_kProject);
        SwapState();

        int spawn = Mathf.Clamp(spawnsPerFrame, 0, MaxSpawnPerFrame);
        if (spawn > 0 && _sourceCount > 0 && _sources != null)
        {
            fluidCompute.SetInt("SpawnCount", spawn);
            fluidCompute.SetTexture(_kSpawn, "ParticlePosAge", _posAge);
            fluidCompute.SetTexture(_kSpawn, "ParticleAux", _aux);
            fluidCompute.SetBuffer(_kSpawn, "Sources", _sources);
            fluidCompute.Dispatch(_kSpawn, (spawn + 63) / 64, 1, 1);
        }

        fluidCompute.SetTexture(_kAdvectParticles, "StateRead", _stateRead);
        fluidCompute.SetTexture(_kAdvectParticles, "ParticlePosAge", _posAge);
        fluidCompute.SetTexture(_kAdvectParticles, "ParticleAux", _aux);
        fluidCompute.Dispatch(_kAdvectParticles, (_particleCapacity + 63) / 64, 1, 1);
    }

    void BindGlobals(float dt)
    {
        float width = _burn != null ? Mathf.Max(_burn.terrainWidth, 1f) : 1f;
        float depth = _burn != null ? Mathf.Max(_burn.terrainDepth, 1f) : 1f;
        float height = _burn != null ? Mathf.Max(_burn.maxHeight, 1f) : 1f;
        ComputeShader shader = fluidCompute;
        shader.SetInt("Res", _res);
        shader.SetInt("Layers", _layers);
        shader.SetFloat("TerrainWidth", width);
        shader.SetFloat("TerrainDepth", depth);
        shader.SetFloat("DomainHeight", height);
        shader.SetFloat("DeltaTime", dt);
        shader.SetFloat("Viscosity", viscosity);
        shader.SetFloat("Buoyancy", buoyancyCoefficient);
        shader.SetFloat("AmbientTemperature", ambientTemperature);
        shader.SetFloat("Dissipation", dissipation);
        shader.SetFloat("WindCoupling", windCoupling);
        shader.SetFloat("HeatSourceStrength", heatSourceStrength);
        shader.SetFloat("MaxSpeed", Mathf.Max(maxSpeed, 0f));
        shader.SetVector("AmbientWind", new Vector4(_ambientWind.x, _ambientWind.y, _ambientWind.z, 0f));
        shader.SetFloat("HeatEnabled", _burn != null && _burn.burnMap != null ? 1f : 0f);
        shader.SetInt("SourceCount", _sourceCount);
        shader.SetInt("SpawnCount", Mathf.Clamp(spawnsPerFrame, 0, MaxSpawnPerFrame));
        shader.SetInt("MaxParticles", _particleCapacity);
        shader.SetFloat("ParticleLifetime", Mathf.Max(particleLifetime, 0.05f));
        shader.SetFloat("ParticleSize", Mathf.Max(particleSize, 0.01f));
        shader.SetInt("FrameSeed", Time.frameCount);
    }

    void ClearGrid(RenderTexture state, RenderTexture pressure)
    {
        fluidCompute.SetTexture(_kInit, "StateWrite", state);
        fluidCompute.SetTexture(_kInit, "PressureWrite", pressure);
        fluidCompute.SetTexture(_kInit, "DivergenceMap", _divergence);
        DispatchGrid(_kInit);
    }

    void DispatchGrid(int kernel)
    {
        fluidCompute.Dispatch(kernel, (_res + 7) / 8, (_res * _layers + 7) / 8, 1);
    }

    void SwapState()
    {
        RenderTexture swap = _stateRead;
        _stateRead = _stateWrite;
        _stateWrite = swap;
    }

    void SwapPressure()
    {
        RenderTexture swap = _pressureRead;
        _pressureRead = _pressureWrite;
        _pressureWrite = swap;
    }

    void UploadSources()
    {
        if (_sources == null || _sourceCount <= 0)
            return;
        _sources.SetData(_sourceData, 0, 0, _sourceCount);
    }

    RenderTexture CreateGrid(int width, int height, RenderTextureFormat format)
    {
        var texture = new RenderTexture(width, height, 0, format, RenderTextureReadWrite.Linear);
        texture.enableRandomWrite = true;
        texture.filterMode = FilterMode.Point;
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.useMipMap = false;
        texture.autoGenerateMips = false;
        texture.Create();
        if (!texture.IsCreated())
            throw new InvalidOperationException($"Failed to create a {width}x{height} {format} fluid texture.");
        return texture;
    }

    void ReleaseResources()
    {
        _ready = false;
        ReleaseTexture(ref _stateA);
        ReleaseTexture(ref _stateB);
        ReleaseTexture(ref _pressureA);
        ReleaseTexture(ref _pressureB);
        ReleaseTexture(ref _divergence);
        ReleaseTexture(ref _posAge);
        ReleaseTexture(ref _aux);
        _stateRead = null;
        _stateWrite = null;
        _pressureRead = null;
        _pressureWrite = null;
        if (_dummyBurn != null)
        {
            Destroy(_dummyBurn);
            _dummyBurn = null;
        }
        if (_sources != null)
        {
            _sources.Release();
            _sources = null;
        }
        if (_particleMaterial != null)
        {
            Destroy(_particleMaterial);
            _particleMaterial = null;
        }
    }

    static void ReleaseTexture(ref RenderTexture texture)
    {
        if (texture == null)
            return;
        texture.Release();
        Destroy(texture);
        texture = null;
    }
}
