using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class FireRenderer : MonoBehaviour
{
    [Header("Fire Visual Settings")]
    [SerializeField] private float maxEmissionRate = 100f;
    [SerializeField] private Color fireColor1 = new Color(1f, 0.8f, 0.2f, 1f);
    [SerializeField] private Color fireColor2 = new Color(1f, 0.4f, 0.1f, 1f);
    [SerializeField] private Color fireColor3 = new Color(0.8f, 0.1f, 0.05f, 1f);
    [SerializeField] private Color smokeColor = new Color(0.2f, 0.2f, 0.2f, 0.5f);
    [SerializeField] private float particleSizeMin = 0.2f;
    [SerializeField] private float particleSizeMax = 0.5f;
    [SerializeField] private float particleLifetime = 1.0f;
    [SerializeField] private float maxFuelDensity = 500f; // kg/m³, adjust based on simulation

    [Header("Performance Settings")]
    [SerializeField] private bool useObjectPooling = true;
    [SerializeField] private int maxPoolSize = 1000;

    [SerializeField] private VisualizationManager visualizationManager;
    [SerializeField] private BurnMapManager burnMapManager;

    private List<GameObject> activeFires = new List<GameObject>();
    private Queue<GameObject> firePool = new Queue<GameObject>();
    private Material fireMaterial;
    private Texture2D fireTexture;
    private float dx, dy;
    private List<float> zLevels;
    private List<float> dz;

    void Awake()
    {
        if (visualizationManager == null)
        {
            visualizationManager = FindObjectOfType<VisualizationManager>();
            if (visualizationManager == null)
            {
                Debug.LogError("VisualizationManager not found in scene.");
            }
        }
        if (burnMapManager == null)
        {
            burnMapManager = FindObjectOfType<BurnMapManager>();
            if (burnMapManager == null)
            {
                Debug.LogError("BurnMapManager not found in scene.");
            }
        }
    }

    void Start()
    {
        InitializeFireMaterial();
        InitializeObjectPool();
        Debug.Log("FireRenderer initialized for particle-based rendering.");
    }

    void OnDestroy()
    {
        ClearFires();
        while (firePool.Count > 0)
        {
            GameObject obj = firePool.Dequeue();
            if (obj != null)
            {
                Destroy(obj);
            }
        }
        if (fireMaterial != null)
        {
            Destroy(fireMaterial);
        }
        if (fireTexture != null)
        {
            Destroy(fireTexture);
        }
    }

    void InitializeFireMaterial()
    {
        fireTexture = GenerateFireTexture();
        fireMaterial = new Material(Shader.Find("Particles/Standard Unlit"));
        fireMaterial.SetTexture("_MainTex", fireTexture);
        fireMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
        fireMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.One);
        fireMaterial.SetInt("_ZWrite", 0);
        fireMaterial.SetFloat("_SoftParticlesNearFadeDistance", 0.0f);
        fireMaterial.SetFloat("_SoftParticlesFarFadeDistance", 1.0f);
    }

    Texture2D GenerateFireTexture()
    {
        int size = 64;
        Texture2D tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        Color[] colors = new Color[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float noise = Mathf.PerlinNoise(x / (float)size * 5f, y / (float)size * 5f);
                colors[x + y * size] = new Color(noise, noise, noise, noise);
            }
        }
        tex.SetPixels(colors);
        tex.Apply();
        return tex;
    }

    GameObject CreateFireObject()
    {
        GameObject fireObj = new GameObject("FireParticleSystem");
        fireObj.transform.SetParent(transform);
        ParticleSystem ps = fireObj.AddComponent<ParticleSystem>();
        ParticleSystemRenderer renderer = fireObj.GetComponent<ParticleSystemRenderer>();

        var main = ps.main;
        main.loop = true;
        main.startLifetime = particleLifetime;
        main.startSpeed = new ParticleSystem.MinMaxCurve(0.5f, 2.0f);
        main.startSize = new ParticleSystem.MinMaxCurve(particleSizeMin, particleSizeMax);
        main.startColor = new ParticleSystem.MinMaxGradient(fireColor3, fireColor1);
        main.maxParticles = 100;
        main.simulationSpace = ParticleSystemSimulationSpace.World;

        var emission = ps.emission;
        emission.enabled = true;
        emission.rateOverTime = 0;

        var shape = ps.shape;
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Box;
        shape.scale = Vector3.one;

        var colorOverLifetime = ps.colorOverLifetime;
        colorOverLifetime.enabled = true;
        Gradient gradient = new Gradient();
        gradient.SetKeys(
            new GradientColorKey[] {
                new GradientColorKey(fireColor1, 0.0f),
                new GradientColorKey(fireColor2, 0.5f),
                new GradientColorKey(fireColor3, 0.8f),
                new GradientColorKey(smokeColor, 1.0f)
            },
            new GradientAlphaKey[] {
                new GradientAlphaKey(1.0f, 0.0f),
                new GradientAlphaKey(1.0f, 0.5f),
                new GradientAlphaKey(0.5f, 0.8f),
                new GradientAlphaKey(0.0f, 1.0f)
            }
        );
        colorOverLifetime.color = gradient;

        var sizeOverLifetime = ps.sizeOverLifetime;
        sizeOverLifetime.enabled = true;
        AnimationCurve sizeCurve = new AnimationCurve(
            new Keyframe(0.0f, 1.0f),
            new Keyframe(0.5f, 1.2f),
            new Keyframe(1.0f, 0.5f)
        );
        sizeOverLifetime.size = new ParticleSystem.MinMaxCurve(1.0f, sizeCurve);

        var velocityOverLifetime = ps.velocityOverLifetime;
        velocityOverLifetime.enabled = true;
        velocityOverLifetime.x = new ParticleSystem.MinMaxCurve(-0.5f, 0.5f);
        velocityOverLifetime.y = new ParticleSystem.MinMaxCurve(1.0f, 2.0f);
        velocityOverLifetime.z = new ParticleSystem.MinMaxCurve(-0.5f, 0.5f);

        var noise = ps.noise;
        noise.enabled = true;
        noise.strength = 0.5f;
        noise.frequency = 1.0f;
        noise.scrollSpeed = 0.5f;

        renderer.renderMode = ParticleSystemRenderMode.Billboard;
        renderer.material = fireMaterial;
        renderer.maxParticleSize = 0.5f;

        return fireObj;
    }

    GameObject GetFireObject()
    {
        if (useObjectPooling && firePool.Count > 0)
        {
            GameObject obj = firePool.Dequeue();
            obj.SetActive(true);
            return obj;
        }
        return CreateFireObject();
    }

    void ReturnFireObject(GameObject fireObj)
    {
        if (useObjectPooling && firePool.Count < maxPoolSize)
        {
            fireObj.SetActive(false);
            firePool.Enqueue(fireObj);
        }
        else
        {
            Destroy(fireObj);
        }
    }

    void InitializeObjectPool()
    {
        if (!useObjectPooling) return;
        for (int i = 0; i < maxPoolSize; i++)
        {
            GameObject fireObj = CreateFireObject();
            fireObj.SetActive(false);
            firePool.Enqueue(fireObj);
        }
        Debug.Log($"Fire object pool initialized with {maxPoolSize} particle systems.");
    }

    public void RenderFires(EpisodeStepData episodeData)
    {
        if (episodeData == null || episodeData.fires == null || episodeData.grid_metadata == null)
        {
            Debug.LogError("Missing fire data or grid metadata.");
            ClearFires();
            return;
        }

        dx = episodeData.grid_metadata.dx;
        dy = episodeData.grid_metadata.dy;
        zLevels = episodeData.grid_metadata.z_levels;
        dz = episodeData.grid_metadata.dz;

        int fireCount = episodeData.fires.Count;
        Debug.Log($"Processing {fireCount} fires.");

        while (activeFires.Count > fireCount)
        {
            ReturnFireObject(activeFires[activeFires.Count - 1]);
            activeFires.RemoveAt(activeFires.Count - 1);
        }
        while (activeFires.Count < fireCount)
        {
            activeFires.Add(GetFireObject());
        }

        List<Vector3> firePositions = new List<Vector3>();
        List<float> fireIntensities = new List<float>();
        List<float> fuelBurnProgress = new List<float>();

        for (int i = 0; i < fireCount; i++)
        {
            FireData fire = episodeData.fires[i];
            Vector3 position = GetUnityPosition(fire);
            float fireDz = FindDzForZLevel(fire.position[2]);
            Vector3 scale = new Vector3(dx, fireDz, dy);

            // Calculate fuel_volume_fraction
            float volume = dx * dy * fireDz; // m³
            float burnProgress = 1.0f - Mathf.Clamp01(fire.fuel_remaining / maxFuelDensity);

            float L = fireDz;
            float Q_max = Mathf.Pow(8.5957f * L, 2.5f);
            float currentQ = fire.intensity;
            float volumeFraction = currentQ > 0 ? Mathf.Clamp01(Mathf.Pow(currentQ / Q_max, 0.4f)) : 0f;

            GameObject fireObj = activeFires[i];
            fireObj.transform.position = position;
            fireObj.transform.localScale = Vector3.one;

            ParticleSystem ps = fireObj.GetComponent<ParticleSystem>();
            if (ps != null)
            {
                var shape = ps.shape;
                shape.shapeType = ParticleSystemShapeType.Box;
                shape.scale = scale;

                var emission = ps.emission;
                emission.rateOverTime = volumeFraction * maxEmissionRate;

                var main = ps.main;
                main.startColor = new ParticleSystem.MinMaxGradient(fireColor3, fireColor1);
                main.startSize = new ParticleSystem.MinMaxCurve(particleSizeMin * (0.5f + volumeFraction), particleSizeMax * (0.5f + volumeFraction));
            }

            firePositions.Add(position);
            fireIntensities.Add(volumeFraction);
            fuelBurnProgress.Add(burnProgress);
            Debug.Log($"Fire {i}: Pos=({fire.position[0]},{fire.position[1]},{fire.position[2]}), Volume={volume}m³, FuelDensity={fire.fuel_remaining}kg/m³, burnProgress={burnProgress}");
        }

        if (burnMapManager != null)
        {
            burnMapManager.UpdateBurnMap(firePositions, fireIntensities, fuelBurnProgress);
        }

        Debug.Log($"[{Time.frameCount}] Rendered {activeFires.Count} fire particle systems.");
    }

    private Vector3 GetUnityPosition(FireData fire)
    {
        float simX = fire.position[0];
        float simY = fire.position[2];
        float simZ = fire.position[1];
        float fireDz = FindDzForZLevel(simY);
        float unityY = simY + (fireDz / 2.0f);
        return new Vector3(simX, unityY, simZ);
    }

    private float FindDzForZLevel(float simY)
    {
        if (zLevels == null || dz == null || zLevels.Count == 0 || dz.Count == 0)
        {
            Debug.LogWarning("Invalid zLevels or dz data, using default height of 1.0f.");
            return 1.0f;
        }

        for (int i = 0; i < zLevels.Count; i++)
        {
            if (Mathf.Abs(zLevels[i] - simY) < 0.01f)
            {
                return i < dz.Count ? dz[i] : 1.0f;
            }
        }

        Debug.LogWarning($"No matching z-level found for simY={simY}, using default dz.");
        return dz.Count > 0 ? dz[0] : 1.0f;
    }

    void ClearFires()
    {
        foreach (GameObject fire in activeFires)
        {
            if (fire != null)
            {
                ReturnFireObject(fire);
            }
        }
        activeFires.Clear();
    }
}