using UnityEngine;
using System.Collections.Generic;
using TMPro;

public class UIAgentResourcesDisplay : MonoBehaviour
{
    [SerializeField] private GameObject agentResourceUIPrefab; // The prefab you created
    [SerializeField] private Transform container; // The parent transform to hold the prefabs

    // Dictionary to hold references to the instantiated UI elements
    private Dictionary<int, UIAgentResourceEntry> agentUIEntries = new Dictionary<int, UIAgentResourceEntry>();

    public void UpdateAgentResources(List<AgentData> agents)
    {
        // First, create or update UI for existing agents
        foreach (var agent in agents)
        {
            if (agentUIEntries.ContainsKey(agent.agent_id))
            {
                // Update existing UI element
                agentUIEntries[agent.agent_id].UpdateResources(agent.resources);
            }
            else
            {
                // Create a new UI element if one doesn't exist
                GameObject newEntryGO = Instantiate(agentResourceUIPrefab, container);
                UIAgentResourceEntry newEntry = newEntryGO.GetComponent<UIAgentResourceEntry>();
                if (newEntry != null)
                {
                    newEntry.Initialize(agent.agent_id, agent.resources);
                    agentUIEntries.Add(agent.agent_id, newEntry);
                }
            }
        }

        // Remove UI elements for agents that are no longer present
        List<int> agentsToRemove = new List<int>();
        foreach (var entry in agentUIEntries)
        {
            if (agents.Find(a => a.agent_id == entry.Key) == null)
            {
                agentsToRemove.Add(entry.Key);
            }
        }

        foreach (int agentId in agentsToRemove)
        {
            Destroy(agentUIEntries[agentId].gameObject);
            agentUIEntries.Remove(agentId);
        }
    }
}