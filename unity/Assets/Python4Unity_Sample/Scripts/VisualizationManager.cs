using UnityEngine;
using System;

public class VisualizationManager : MonoBehaviour
{
    public SimulationDataLoader dataLoader;
    public TerrainGenerator terrainGenerator;
    public AgentRenderer agentRenderer;
    public FireRenderer fireRenderer;
    public PlumeRenderer plumeRenderer;
    public XRCameraController cameraController;
    public UIMissionControlManager uIMissionControlManager;

    // Event to signal when vegetation is ready
    public event Action OnVegetationReady;

    void Start()
    {
        if (dataLoader == null)
        {
            Debug.LogError("SimulationDataLoader is not assigned.");
            return;
        }
        if (terrainGenerator == null)
        {
            Debug.LogError("TerrainGenerator is not assigned.");
            return;
        }
        if (agentRenderer == null)
        {
            Debug.LogError("AgentRenderer is not assigned.");
            return;
        }
        if (fireRenderer == null)
        {
            Debug.LogError("FireRenderer is not assigned.");
            return;
        }
        if (plumeRenderer == null)
        {
            Debug.LogError("PlumeRenderer is not assigned.");
            return;
        }
        if (uIMissionControlManager == null)
        {
            Debug.LogError("UIMissionControlManager is not assigned.");
            return;
        }
        if (cameraController == null)
        {
            Debug.LogError("XRCameraController is not assigned.");
            return;
        }

        dataLoader.OnInitialEnvironmentDataAvailable += OnInitialDataLoaded;
        dataLoader.OnNewEpisodeStepDataAvailable += OnEpisodeStepDataLoaded;

        agentRenderer.OnAgentSelected += cameraController.SetAgent;
        Debug.Log("Subscribed to AgentRenderer.OnAgentSelected");

        if (dataLoader.CurrentInitialEnvironmentData != null)
        {
            OnInitialDataLoaded(dataLoader.CurrentInitialEnvironmentData);
        }
        if (dataLoader.CurrentEpisodeStepData != null)
        {
            OnEpisodeStepDataLoaded(dataLoader.CurrentEpisodeStepData);
        }
    }

    void OnInitialDataLoaded(InitialEnvironmentData initialData)
    {
        Debug.Log($"[{Time.frameCount}] Starting terrain and vegetation generation...");
        terrainGenerator.GenerateTerrain(initialData);
        terrainGenerator.AddVegetation(initialData);
        GameObject[] vegetation = GameObject.FindGameObjectsWithTag("Vegetation");
        Debug.Log($"[{Time.frameCount}] Vegetation generation completed. Found {vegetation.Length} vegetation objects.");
        agentRenderer.SetInitialEnvironmentData(initialData);
        uIMissionControlManager.SetInitialEnvironmentData(initialData);
        // Signal that vegetation is ready
        OnVegetationReady?.Invoke();
    }

    void OnEpisodeStepDataLoaded(EpisodeStepData episodeData)
    {
        Debug.Log($"[{Time.frameCount}] Starting episode step rendering...");
        GameObject[] vegetation = GameObject.FindGameObjectsWithTag("Vegetation");
        Debug.Log($"[{Time.frameCount}] Before RenderFires: Found {vegetation.Length} vegetation objects.");
        agentRenderer.RenderAgents(episodeData);
        fireRenderer.RenderFires(episodeData);
        plumeRenderer.RenderPlumes(episodeData);
        uIMissionControlManager.UpdateUI(episodeData);
        Debug.Log($"[{Time.frameCount}] Episode step rendering completed.");
    }
}
