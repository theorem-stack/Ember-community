using UnityEngine;
using System.Collections.Generic;

public class BurnMapManager : MonoBehaviour
{
    public ComputeShader burnComputeShader;
    public int mapResolution = 256;
    public float burnRadius = 5.0f;
    public float terrainWidth = 256;
    public float terrainDepth = 256;
    public float maxHeight = 100.0f;

    [HideInInspector]
    public RenderTexture burnMap;

    private int updateBurnMapKernel;
    private ComputeBuffer fireDataBuffer;
    private FireDataGPU[] fireDataArray;

    private struct FireDataGPU
    {
        public Vector3 position;
        public float intensity;
        public float burnProgress; // Renamed to burnProgress
    }

    void Start()
    {
        if (burnComputeShader == null)
        {
            Debug.LogError("Burn Compute Shader is not assigned!");
            return;
        }

        burnMap = new RenderTexture(mapResolution, mapResolution, 0, RenderTextureFormat.ARGBFloat);
        burnMap.enableRandomWrite = true;
        burnMap.Create();

        updateBurnMapKernel = burnComputeShader.FindKernel("UpdateBurnMap");
        burnComputeShader.SetTexture(updateBurnMapKernel, "BurnMap", burnMap);
        burnComputeShader.SetFloat("BurnRadius", burnRadius);
        burnComputeShader.SetInt("MapResolution", mapResolution);
        burnComputeShader.SetFloat("TerrainWidth", terrainWidth);
        burnComputeShader.SetFloat("TerrainDepth", terrainDepth);
        burnComputeShader.SetFloat("MaxHeight", maxHeight);
        Shader.SetGlobalTexture("_BurnMap", burnMap);
    }

    void OnDestroy()
    {
        if (burnMap != null) burnMap.Release();
        if (fireDataBuffer != null) fireDataBuffer.Release();
    }

    public void InitializeBurnMap(InitialEnvironmentData initialData)
    {
        if (initialData?.fuel_map == null)
        {
            Debug.LogError("InitialEnvironmentData or fuel_map is null.");
            return;
        }

        int rows = initialData.fuel_map.Count;
        int cols = rows > 0 ? initialData.fuel_map[0].Count : 0;

        Texture2D initialFuelTex = new Texture2D(cols, rows, TextureFormat.ARGB32, false);
        Color[] pixels = new Color[cols * rows];

        // Fill the blue channel with 1.0 (unburnt). This now represents burnProgress = 0.
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                // We are now storing burn progress in the blue channel, so 0.0 means unburnt.
                pixels[y * cols + x] = new Color(0, 0, 0.0f, 1.0f);
            }
        }

        initialFuelTex.SetPixels(pixels);
        initialFuelTex.Apply();

        Graphics.Blit(initialFuelTex, burnMap);

        Destroy(initialFuelTex);
        Debug.Log("BurnMap initialized with unburnt fuel data.");
    }

    public void UpdateBurnMap(List<Vector3> firePositions, List<float> fireIntensities, List<float> fuelBurnProgress)
    {
        if (firePositions == null || fireIntensities == null || fuelBurnProgress == null ||
            firePositions.Count == 0 || fireIntensities.Count == 0 || fuelBurnProgress.Count == 0 ||
            firePositions.Count != fireIntensities.Count || fireIntensities.Count != fuelBurnProgress.Count)
        {
            Debug.LogWarning($"Invalid fire data: {firePositions?.Count ?? 0} positions, {fireIntensities?.Count ?? 0} intensities, {fuelBurnProgress?.Count ?? 0} burn progress values");
            return;
        }

        Debug.Log($"[{Time.frameCount}] Updating burn map with {firePositions.Count} fires");
        for (int i = 0; i < firePositions.Count; i++)
        {
            Debug.Log($"Fire {i}: Pos={firePositions[i]}, Height={firePositions[i].y}, Intensity={fireIntensities[i]}, BurnProgress={fuelBurnProgress[i]}");
        }

        if (fireDataBuffer == null || fireDataBuffer.count < firePositions.Count)
        {
            if (fireDataBuffer != null) fireDataBuffer.Release();
            // The size of the struct is now Vector3 (12 bytes) + float (4 bytes) + float (4 bytes) = 20 bytes
            fireDataBuffer = new ComputeBuffer(firePositions.Count, sizeof(float) * 5);
        }

        fireDataArray = new FireDataGPU[firePositions.Count];
        for (int i = 0; i < firePositions.Count; i++)
        {
            fireDataArray[i].position = firePositions[i];
            fireDataArray[i].intensity = fireIntensities[i];
            fireDataArray[i].burnProgress = fuelBurnProgress[i]; // Use the new burnProgress value
        }

        fireDataBuffer.SetData(fireDataArray);
        burnComputeShader.SetBuffer(updateBurnMapKernel, "FireData", fireDataBuffer);
        burnComputeShader.SetInt("NumFires", firePositions.Count);
        burnComputeShader.SetFloat("DeltaTime", Time.deltaTime);
        burnComputeShader.Dispatch(updateBurnMapKernel, mapResolution / 8, mapResolution / 8, 1);
    }
}