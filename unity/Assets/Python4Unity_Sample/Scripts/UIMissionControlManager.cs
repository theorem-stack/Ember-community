using UnityEngine;
using System.Collections.Generic;

public class UIMissionControlManager : MonoBehaviour
{
    [SerializeField] private UIGridController gridController;
    [SerializeField] private UIMissionLogger missionLogger;
    [SerializeField] private UIAgentResourcesDisplay agentResourcesDisplay;

    public void UpdateUI(EpisodeStepData data)
    {
        if (data == null)
        {
            Debug.LogWarning("Cannot update UI: EpisodeStepData is null.");
            return;
        }

        // Update the grid controller if the reference exists
        if (gridController != null)
        {
            gridController.UpdateGrid(data);
        }
        else
        {
            Debug.LogError("Grid Controller reference is missing!");
        }

        // Update the mission logger if the reference exists
        if (missionLogger != null)
        {
            missionLogger.LogAgentUpdates(data.agents);
        }
        else
        {
            Debug.LogError("Mission Logger reference is missing!");
        }

        // Update the agent resources display if the reference exists
        if (agentResourcesDisplay != null)
        {
            agentResourcesDisplay.UpdateAgentResources(data.agents);
        }
        else
        {
            Debug.LogError("Agent Resources Display reference is missing!");
        }
    }

    public void SetInitialEnvironmentData(InitialEnvironmentData initialEnvironmentData)
    {
        if (gridController != null)
        {
            gridController.SetInitialEnvironmentData(initialEnvironmentData);
        }
        else
        {
            Debug.LogError("Grid Controller reference is missing!");
        }
    }
}