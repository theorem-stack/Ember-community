using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class UIGridController : MonoBehaviour
{
    private Vector2 gridSize = new Vector2(50, 50);
    [SerializeField] private float gridDotRadiusMultiplier = 0.01f;
    [SerializeField] private float agentRadiusMultiplier = 0.5f;
    [SerializeField] private float targetRadiusMultiplier = 0.7f;
    [SerializeField] private float pathDotRadiusMultiplier = 0.2f;
    [SerializeField] private Color backgroundColor = new Color(0.1f, 0.15f, 0.2f, 0.5f);
    [SerializeField] private float glowIntensity = 1.5f;
    [SerializeField] private float pulseSpeed = 2.0f;
    [SerializeField] private Color heatGradientStart = new Color(0f, 0.5f, 1f, 0.3f);
    [SerializeField] private Color heatGradientEnd = new Color(1f, 0f, 0f, 0.8f);
    [SerializeField] private Color burnBoxColor = new Color(1f, 0.2f, 0f, 0.2f); // New: Burn box tint
    private RawImage rawImage;
    private Texture2D heatFluxTexture;
    private Texture2D burnBoxTexture; // New: Texture for burn_box_mask
    private ComputeBuffer agentBuffer;
    private List<Vector4> agentData;
    private ComputeBuffer targetBuffer;
    private List<Vector4> targetData;
    private int targetCount;
    private ComputeBuffer pathBuffer;
    private List<Vector4> pathData;
    private int pathPointCount;
    private List<bool[]> burn_box_mask;

    void Start()
    {
        rawImage = GetComponent<RawImage>();
        if (rawImage == null)
        {
            Debug.LogError("UIGridController requires a RawImage component!");
            return;
        }

        rawImage.material = new Material(Shader.Find("Custom/UIGridShader"));
        UpdateGridProperties();
#if UNITY_EDITOR
        Debug.Log("UIGridController initialized with material: " + rawImage.material.name);
#endif
    }

    void OnValidate()
    {
        if (rawImage != null && rawImage.material != null)
        {
            UpdateGridProperties();
        }
    }

    public void SetInitialEnvironmentData(InitialEnvironmentData initialEnvironmentData)
    {
        burn_box_mask = initialEnvironmentData.burn_box_mask;
    }

    public void UpdateGrid(EpisodeStepData data)
    {
        if (data == null || data.grid_metadata == null)
        {
            Debug.LogWarning("Cannot update grid: EpisodeStepData or grid_metadata is null.");
            return;
        }

        GridMetadata metadata = data.grid_metadata;
        int gridWidth = metadata.agent_obs_x;
        int gridHeight = metadata.agent_obs_y;

        if (gridWidth <= 0 || gridHeight <= 0)
        {
            Debug.LogWarning($"Invalid grid dimensions: {gridWidth}x{gridHeight}");
            return;
        }

        gridSize.x = gridWidth;
        gridSize.y = gridHeight;

        agentData = new List<Vector4>();
        targetData = new List<Vector4>();
        pathData = new List<Vector4>();
        if (data.agents != null && data.agents.Count > 0)
        {
            foreach (var agent in data.agents)
            {
                if (agent._ai != null && agent._ai.position != null && agent._ai.position.Length >= 2)
                {
                    Vector2 agentPos = new Vector2(agent._ai.position[0], agent._ai.position[1]);
                    if (agentPos.x >= 0 && agentPos.x < gridWidth && agentPos.y >= 0 && agentPos.y < gridHeight)
                    {
                        Vector2 normalizedPos = new Vector2((agentPos.x + 0.5f) / gridWidth, (agentPos.y + 0.5f) / gridHeight);
                        Color color = GenerateColorFromAgentId(agent.agent_id);
                        agentData.Add(new Vector4(normalizedPos.x, normalizedPos.y, color.r, color.g));
                        agentData.Add(new Vector4(0, 0, color.b, 0));
                    }
                }

                if (agent.intent != null)
                {
                    Vector2 targetPos = new Vector2(agent.intent.x, agent.intent.y);
                    int targetType = agent.intent.type;
                    Vector2 normalizedTargetPos = new Vector2((targetPos.x + 0.5f) / gridWidth, (targetPos.y + 0.5f) / gridHeight);
                    Color targetColor = GenerateColorFromTargetType(targetType);
                    targetData.Add(new Vector4(normalizedTargetPos.x, normalizedTargetPos.y, targetColor.r, targetColor.g));
                    targetData.Add(new Vector4(0, 0, targetColor.b, 0));
                }

                if (agent._ai != null && agent._ai.path != null && agent._ai.path.Count > 0)
                {
                    Color pathColor = GenerateColorFromAgentId(agent.agent_id);
                    foreach (var pathPoint in agent._ai.path)
                    {
                        if (pathPoint.Length >= 2)
                        {
                            Vector2 pathPos = new Vector2(pathPoint[0], pathPoint[1]);
                            Vector2 normalizedPathPos = new Vector2((pathPos.x + 0.5f) / gridWidth, (pathPos.y + 0.5f) / gridHeight);
                            pathData.Add(new Vector4(normalizedPathPos.x, normalizedPathPos.y, pathColor.r, pathColor.g));
                            pathData.Add(new Vector4(0, 0, pathColor.b, 0));
                        }
                    }
                }
            }
        }

        targetCount = targetData.Count / 2;
        pathPointCount = pathData.Count / 2;

#if UNITY_EDITOR
        Debug.Log($"Processed {agentData.Count / 2} agents, {targetCount} targets, and {pathPointCount} path points");
#endif

        UpdateAgentBuffer();
        UpdateTargetBuffer();
        UpdatePathBuffer();
        UpdateHeatFluxTexture(data.agent_obs_heat_flux, gridWidth, gridHeight);
        UpdateBurnBoxTexture(burn_box_mask, gridWidth, gridHeight); // New: Process burn_box_mask
        UpdateGridProperties();
#if UNITY_EDITOR
        Debug.Log($"UI Grid updated: {gridSize.x}x{gridSize.y}");
#endif
    }

    private Color GenerateColorFromAgentId(int agentId)
    {
        const float goldenRatioConjugate = 0.61803398875f;
        float hue = (agentId * goldenRatioConjugate) % 1.0f;
        float saturation = 0.8f;
        float value = 0.95f;
        return Color.HSVToRGB(hue, saturation, value);
    }

    private Color GenerateColorFromTargetType(int targetType)
    {
        const float goldenRatioConjugate = 0.61803398875f;
        float hue = (targetType * goldenRatioConjugate) % 1.0f;
        float saturation = 0.9f;
        float value = 1.0f;
        return Color.HSVToRGB(hue, saturation, value);
    }

    private void UpdateAgentBuffer()
    {
        if (agentBuffer != null)
        {
            agentBuffer.Release();
            agentBuffer = null;
        }
        agentBuffer = new ComputeBuffer(agentData.Count > 0 ? agentData.Count : 1, sizeof(float) * 4);
        if (agentData.Count > 0)
        {
            agentBuffer.SetData(agentData);
        }
        rawImage.material.SetBuffer("_AgentData", agentBuffer);
        rawImage.material.SetInt("_AgentCount", agentData.Count / 2);
    }

    private void UpdatePathBuffer()
    {
        if (pathBuffer != null)
        {
            pathBuffer.Release();
            pathBuffer = null;
        }
        pathBuffer = new ComputeBuffer(pathData.Count > 0 ? pathData.Count : 1, sizeof(float) * 4);
        if (pathData.Count > 0)
        {
            pathBuffer.SetData(pathData);
        }
        rawImage.material.SetBuffer("_PathData", pathBuffer);
        rawImage.material.SetInt("_PathCount", pathData.Count / 2);
    }

    private void UpdateTargetBuffer()
    {
        if (targetBuffer != null)
        {
            targetBuffer.Release();
            targetBuffer = null;
        }
        targetBuffer = new ComputeBuffer(targetData.Count > 0 ? targetData.Count : 1, sizeof(float) * 4);
        if (targetData.Count > 0)
        {
            targetBuffer.SetData(targetData);
        }
        rawImage.material.SetBuffer("_TargetData", targetBuffer);
        rawImage.material.SetInt("_TargetCount", targetData.Count / 2);
    }

    private void UpdateHeatFluxTexture(List<float[]> heatFlux, int width, int height)
    {
        if (heatFlux == null || heatFlux.Count == 0)
        {
            Debug.LogWarning("Heat flux data is empty or null.");
            return;
        }

        if (heatFlux.Count != height || (heatFlux.Count > 0 && heatFlux[0].Length != width))
        {
            Debug.LogWarning($"Heat flux dimensions mismatch: Expected {width}x{height}, got {heatFlux[0].Length}x{heatFlux.Count}");
            return;
        }

        if (heatFluxTexture == null || heatFluxTexture.width != width || heatFluxTexture.height != height)
        {
            if (heatFluxTexture != null) Destroy(heatFluxTexture);
            heatFluxTexture = new Texture2D(width, height, TextureFormat.RFloat, false);
            heatFluxTexture.filterMode = FilterMode.Point;
            heatFluxTexture.wrapMode = TextureWrapMode.Clamp;
        }
        else
        {
            heatFluxTexture.SetPixels(new Color[width * height]);
        }

        float maxHeat = 0f;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float val = heatFlux[x][y];
                maxHeat = Mathf.Max(maxHeat, Mathf.Abs(val));
            }
        }
        maxHeat = Mathf.Max(maxHeat, 0.0001f);

#if UNITY_EDITOR
        Debug.Log($"Max heat flux value: {maxHeat}");
#endif

        Color[] colors = new Color[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float heat = heatFlux[x][y];
                float normalizedHeat = heat / maxHeat;
                colors[y * width + x] = new Color(normalizedHeat, 0f, 0f, 1f);
            }
        }
        heatFluxTexture.SetPixels(colors);
        heatFluxTexture.Apply();
        rawImage.material.SetTexture("_HeatFluxTex", heatFluxTexture);
#if UNITY_EDITOR
        Debug.Log($"Heat flux texture updated: {width}x{height}, Format: {heatFluxTexture.format}");
#endif
    }

    private void UpdateBurnBoxTexture(List<bool[]> burnBoxMask, int width, int height)
    {
        if (burnBoxMask == null || burnBoxMask.Count == 0)
        {
            Debug.LogWarning("Burn box mask data is empty or null.");
            return;
        }

        if (burnBoxMask.Count != height || (burnBoxMask.Count > 0 && burnBoxMask[0].Length != width))
        {
            Debug.LogWarning($"Burn box mask dimensions mismatch: Expected {width}x{height}, got {burnBoxMask[0].Length}x{burnBoxMask.Count}");
            return;
        }

        if (burnBoxTexture == null || burnBoxTexture.width != width || burnBoxTexture.height != height)
        {
            if (burnBoxTexture != null) Destroy(burnBoxTexture);
            burnBoxTexture = new Texture2D(width, height, TextureFormat.RFloat, false);
            burnBoxTexture.filterMode = FilterMode.Point;
            burnBoxTexture.wrapMode = TextureWrapMode.Clamp;
        }
        else
        {
            burnBoxTexture.SetPixels(new Color[width * height]);
        }

        Color[] colors = new Color[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float value = burnBoxMask[x][y] ? 1f : 0f;
                colors[y * width + x] = new Color(value, 0f, 0f, 1f);
            }
        }
        burnBoxTexture.SetPixels(colors);
        burnBoxTexture.Apply();
        rawImage.material.SetTexture("_BurnBoxTex", burnBoxTexture);
#if UNITY_EDITOR
        Debug.Log($"Burn box texture updated: {width}x{height}, Format: {burnBoxTexture.format}");
#endif
    }

    private void UpdateGridProperties()
    {
        if (rawImage == null || rawImage.material == null)
        {
            Debug.LogError("RawImage or material is null!");
            return;
        }

        float gridDotRadius = gridDotRadiusMultiplier / Mathf.Max(gridSize.x, gridSize.y);
        float agentRadius = agentRadiusMultiplier / Mathf.Max(gridSize.x, gridSize.y);
        float targetRadius = targetRadiusMultiplier / Mathf.Max(gridSize.x, gridSize.y);
        float pathDotRadius = pathDotRadiusMultiplier / Mathf.Max(gridSize.x, gridSize.y);

        rawImage.material.SetVector("_GridSize", gridSize);
        rawImage.material.SetFloat("_GridDotRadius", gridDotRadius);
        rawImage.material.SetFloat("_AgentRadius", agentRadius);
        rawImage.material.SetFloat("_TargetRadius", targetRadius);
        rawImage.material.SetFloat("_PathDotRadius", pathDotRadius);
        rawImage.material.SetColor("_BackgroundColor", backgroundColor);
        rawImage.material.SetFloat("_GlowIntensity", glowIntensity);
        rawImage.material.SetFloat("_PulseSpeed", pulseSpeed);
        rawImage.material.SetColor("_HeatGradientStart", heatGradientStart);
        rawImage.material.SetColor("_HeatGradientEnd", heatGradientEnd);
        rawImage.material.SetColor("_BurnBoxColor", burnBoxColor); // New: Set burn box color
    }

    void OnDestroy()
    {
        if (heatFluxTexture != null)
        {
            Destroy(heatFluxTexture);
        }
        if (burnBoxTexture != null) // New: Clean up burn box texture
        {
            Destroy(burnBoxTexture);
        }
        if (agentBuffer != null)
        {
            agentBuffer.Release();
        }
        if (targetBuffer != null)
        {
            targetBuffer.Release();
        }
        if (pathBuffer != null)
        {
            pathBuffer.Release();
        }
        if (rawImage != null && rawImage.material != null)
        {
            Destroy(rawImage.material);
        }
    }
}