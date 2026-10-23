using UnityEngine;
using System.Collections.Generic;

public class PlumeRenderer : MonoBehaviour
{
    [Header("Plume Visual Settings")]
    [SerializeField] private float maxEmissionRate = 200f; // Max particles per second
    [SerializeField] private float particleSizeMin = 1.0f; // Min particle size
    [SerializeField] private float particleSizeMax = 3.0f; // Max particle size
    [SerializeField] private float particleLifetime = 20.0f; // Max lifetime
    [SerializeField] private float maxPlumeHeight = 100f; // Max h_sub
    [SerializeField] private float maxTotalHeat = 1000000f; // Max total_heat in kW, adjusted for CFD

    [Header("Performance Settings")]
    [SerializeField] private bool useObjectPooling = true;
    [SerializeField] private int maxPoolSize = 500;

    [Header("Particle System Core Settings")]
    [SerializeField] private int maxParticlesPerSystem = 1000;
    [SerializeField] private bool loopParticles = true;
    [SerializeField] private bool playOnAwake = false;

    [Header("Shape Settings")]
    [SerializeField] private ParticleSystemShapeType shapeType = ParticleSystemShapeType.Cone;
    [SerializeField] private float shapeRadiusMin = 1f;
    [SerializeField] private float shapeRadiusMax = 5f;
    [SerializeField] private float shapeAngle = 10f; // Cone angle for plume base
    [SerializeField] private Mesh plumeMeshShape; // New field for custom mesh shape

    [Header("Velocity Settings")]
    [SerializeField] private AnimationCurve verticalVelocityCurve; // Vertical: starts fast, slows down
    [SerializeField] private AnimationCurve horizontalVelocityCurve; // Horizontal: starts slow, speeds up

    [Header("Color Over Lifetime")]
    [SerializeField] private Gradient colorOverLifetimeGradient;

    [Header("Turbulence Settings")]
    [SerializeField] private float turbulenceScale = 0.1f; // Scales k for velocity perturbations
    [SerializeField] private float noiseStrength = 1.0f; // Particle system noise strength
    [SerializeField] private float noiseFrequency = 0.5f;
    [SerializeField] private float noiseScrollSpeed = 0.5f;

    [Header("Texture Settings")]
    [SerializeField] private int plumeTextureSize = 64;

    [SerializeField] private VisualizationManager visualizationManager;

    [Header("Fluid Coupling")]
    [Tooltip("When a HeadsetFluidSimulation in this scene is simulating, plume particles are advected on the GPU by that velocity field and these CPU particle systems stay off.")]
    [SerializeField] private HeadsetFluidSimulation headsetFluid;

    private List<GameObject> activePlumes = new List<GameObject>();
    private Queue<GameObject> plumePool = new Queue<GameObject>();
    private Material plumeMaterial;
    private Texture2D plumeTexture;

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

        if (colorOverLifetimeGradient == null)
        {
            colorOverLifetimeGradient = new Gradient();
            colorOverLifetimeGradient.SetKeys(
                new GradientColorKey[] {
                    new GradientColorKey(new Color(1f, 0f, 0f), 0.0f), // Red (low heat)
                    new GradientColorKey(new Color(1f, 1f, 0f), 0.5f), // Yellow (medium heat)
                    new GradientColorKey(new Color(1f, 1f, 1f), 1.0f)  // White (high heat)
                },
                new GradientAlphaKey[] {
                    new GradientAlphaKey(1.0f, 0.0f),
                    new GradientAlphaKey(0.7f, 0.5f),
                    new GradientAlphaKey(0.0f, 1.0f)
                }
            );
        }

        if (verticalVelocityCurve == null)
        {
            verticalVelocityCurve = new AnimationCurve(
                new Keyframe(0f, 1f), // Starts at full vertical velocity
                new Keyframe(1f, 0f)  // Ends at zero
            );
        }

        if (horizontalVelocityCurve == null)
        {
            horizontalVelocityCurve = new AnimationCurve(
                new Keyframe(0f, 0f), // Starts at zero horizontal velocity
                new Keyframe(1f, 1f)  // Ends at full horizontal velocity
            );
        }
    }

    void Start()
    {
        if (FluidOwnsPlumes())
        {
            Debug.Log("PlumeRenderer is using the headset fluid velocity field. CPU plume particles stay off.");
            return;
        }
        InitializePlumeMaterial();
        InitializeObjectPool();
        Debug.Log("PlumeRenderer initialized for particle-based rendering.");
    }

    void OnDestroy()
    {
        ClearPlumes();
        while (plumePool.Count > 0)
        {
            GameObject obj = plumePool.Dequeue();
            if (obj != null) Destroy(obj);
        }
        if (plumeMaterial != null) Destroy(plumeMaterial);
        if (plumeTexture != null) Destroy(plumeTexture);
    }

    void InitializePlumeMaterial()
    {
        plumeTexture = GeneratePlumeTexture();
        plumeMaterial = new Material(Shader.Find("Particles/Standard Unlit"));
        plumeMaterial.SetTexture("_MainTex", plumeTexture);
        plumeMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        plumeMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.One);
        plumeMaterial.SetInt("_ZWrite", 0);
    }

    Texture2D GeneratePlumeTexture()
    {
        int size = plumeTextureSize;
        Texture2D tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        Color[] colors = new Color[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float dx = x / (float)size - 0.5f;
                float dy = y / (float)size - 0.5f;
                float dist = Mathf.Sqrt(dx * dx + dy * dy);
                float alpha = Mathf.Clamp01(1.0f - dist * 2f);
                colors[x + y * size] = new Color(1f, 1f, 1f, alpha);
            }
        }
        tex.SetPixels(colors);
        tex.Apply();
        return tex;
    }

    GameObject CreatePlumeObject()
    {
        GameObject plumeObj = new GameObject("PlumeParticleSystem");
        plumeObj.transform.SetParent(transform);
        ParticleSystem ps = plumeObj.AddComponent<ParticleSystem>();
        ParticleSystemRenderer renderer = ps.GetComponent<ParticleSystemRenderer>();

        var main = ps.main;
        main.maxParticles = maxParticlesPerSystem;
        main.loop = loopParticles;
        main.playOnAwake = playOnAwake;
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.startLifetime = particleLifetime;
        main.startSpeed = 0f;

        var emission = ps.emission;
        emission.enabled = true;
        emission.rateOverTime = 0;

        var shape = ps.shape;
        shape.enabled = true;
        shape.shapeType = shapeType;
        shape.radius = shapeRadiusMin;
        shape.angle = shapeAngle;

        var velocityOverLifetime = ps.velocityOverLifetime;
        velocityOverLifetime.enabled = true;
        velocityOverLifetime.space = ParticleSystemSimulationSpace.Local;

        var colorOverLifetime = ps.colorOverLifetime;
        colorOverLifetime.enabled = true;
        colorOverLifetime.color = new ParticleSystem.MinMaxGradient(colorOverLifetimeGradient);

        var sizeOverLifetime = ps.sizeOverLifetime;
        sizeOverLifetime.enabled = true;
        sizeOverLifetime.size = new ParticleSystem.MinMaxCurve(1.0f, new AnimationCurve(
            new Keyframe(0.0f, particleSizeMin / particleSizeMax),
            new Keyframe(1.0f, 1.0f)
        ));

        var noise = ps.noise;
        noise.enabled = true;
        noise.strength = noiseStrength;
        noise.frequency = noiseFrequency;
        noise.scrollSpeed = noiseScrollSpeed;

        renderer.renderMode = ParticleSystemRenderMode.Billboard;
        renderer.material = plumeMaterial;

        return plumeObj;
    }

    GameObject GetPlumeObject()
    {
        if (useObjectPooling && plumePool.Count > 0)
        {
            GameObject obj = plumePool.Dequeue();
            obj.SetActive(true);
            return obj;
        }
        return CreatePlumeObject();
    }

    void ReturnPlumeObject(GameObject plumeObj)
    {
        if (useObjectPooling && plumePool.Count < maxPoolSize)
        {
            plumeObj.SetActive(false);
            plumePool.Enqueue(plumeObj);
        }
        else
        {
            Destroy(plumeObj);
        }
    }

    void InitializeObjectPool()
    {
        if (!useObjectPooling) return;
        for (int i = 0; i < maxPoolSize; i++)
        {
            GameObject plumeObj = CreatePlumeObject();
            plumeObj.SetActive(false);
            plumePool.Enqueue(plumeObj);
        }
        Debug.Log($"Plume object pool initialized with {maxPoolSize} particle systems.");
    }

    public void RenderPlumes(EpisodeStepData episodeData)
    {
        if (FluidOwnsPlumes())
        {
            headsetFluid.SetPlumeSources(episodeData != null ? episodeData.plumes : null);
            ClearPlumes();
            return;
        }

        if (episodeData == null || episodeData.plumes == null)
        {
            Debug.LogError("Missing plume data.");
            ClearPlumes();
            return;
        }

        int plumeCount = episodeData.plumes.Count;
        Debug.Log($"Processing {plumeCount} plumes.");

        while (activePlumes.Count > plumeCount)
        {
            ReturnPlumeObject(activePlumes[activePlumes.Count - 1]);
            activePlumes.RemoveAt(activePlumes.Count - 1);
        }
        while (activePlumes.Count < plumeCount)
        {
            activePlumes.Add(GetPlumeObject());
        }

        for (int i = 0; i < plumeCount; i++)
        {
            PlumeData plume = episodeData.plumes[i];
            if (plume == null || plume.origin_position == null || plume.origin_position.Length != 3)
            {
                Debug.LogWarning($"Invalid plume data at index {i}. Skipping.");
                continue;
            }

            GameObject plumeObj = activePlumes[i];
            ParticleSystem ps = plumeObj.GetComponent<ParticleSystem>();
            if (ps == null) continue;

            Vector3 position = new Vector3(plume.origin_position[0], plume.origin_position[2], plume.origin_position[1]);
            plumeObj.transform.position = position;

            float heatNormalized = Mathf.Clamp01(plume.total_heat / maxTotalHeat);
            float heightNormalized = Mathf.Clamp01(plume.h_sub / maxPlumeHeight);
            float turbulence = Mathf.Clamp(plume.k, 0f, 10f);

            var main = ps.main;
            var emission = ps.emission;
            var shape = ps.shape;
            var velocityOverLifetime = ps.velocityOverLifetime;
            var noise = ps.noise;
            var colorOverLifetime = ps.colorOverLifetime;

            emission.rateOverTime = heatNormalized * maxEmissionRate;
            main.startSize = new ParticleSystem.MinMaxCurve(particleSizeMin + (particleSizeMax - particleSizeMin) * heatNormalized);
            main.startLifetime = particleLifetime * heightNormalized;

            shape.radius = Mathf.Lerp(shapeRadiusMin, shapeRadiusMax, plume.ell / (plume.ell + 1f));

            Vector3 plumeVelocity = new Vector3(plume.plume_u, plume.plume_w, plume.plume_v);

            velocityOverLifetime.enabled = true;
            // CHANGE HERE: Set simulation space to World
            velocityOverLifetime.space = ParticleSystemSimulationSpace.World;

            // Apply the velocity curves to the world axes
            velocityOverLifetime.y = new ParticleSystem.MinMaxCurve(plumeVelocity.y, verticalVelocityCurve);
            velocityOverLifetime.x = new ParticleSystem.MinMaxCurve(plumeVelocity.x, horizontalVelocityCurve);
            velocityOverLifetime.z = new ParticleSystem.MinMaxCurve(plumeVelocity.z, horizontalVelocityCurve);

            // Rotate the entire plume object to handle the horizontal tilt, but only around the Y axis
            Vector3 horizontalVelocity = new Vector3(plumeVelocity.x, 0, plumeVelocity.z);
            if (horizontalVelocity != Vector3.zero)
            {
                plumeObj.transform.rotation = Quaternion.LookRotation(horizontalVelocity, Vector3.up);
            }
            else
            {
                plumeObj.transform.rotation = Quaternion.identity;
            }

            noise.enabled = true;
            noise.strength = turbulence * turbulenceScale;

            colorOverLifetime.color = new ParticleSystem.MinMaxGradient(colorOverLifetimeGradient);

            main.startLifetime = particleLifetime * heightNormalized;
            main.startSize = new ParticleSystem.MinMaxCurve(particleSizeMin + (particleSizeMax - particleSizeMin) * heatNormalized);

            if (!ps.isPlaying)
            {
                ps.Play();
            }
        }
        Debug.Log($"[{Time.frameCount}] Rendered {activePlumes.Count} plume particle systems.");
    }

    void ClearPlumes()
    {
        foreach (GameObject plume in activePlumes)
        {
            if (plume != null)
            {
                ReturnPlumeObject(plume);
            }
        }
        activePlumes.Clear();
    }

    bool FluidOwnsPlumes()
    {
        if (headsetFluid == null)
        {
            headsetFluid = FindObjectOfType<HeadsetFluidSimulation>();
        }
        return headsetFluid != null && headsetFluid.WillDrivePlumes;
    }
}