using UnityEngine;
using TMPro;

public class UIAgentResourceEntry : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI agentIdText;
    [SerializeField] private TextMeshProUGUI waterText;
    [SerializeField] private TextMeshProUGUI retardantText;
    [SerializeField] private TextMeshProUGUI airSupportText;

    public void Initialize(int agentId, ResourcesData resources)
    {
        agentIdText.text = $"Agent {agentId}:";
        UpdateResources(resources);
    }

    public void UpdateResources(ResourcesData resources)
    {
        waterText.text = $"Water: {resources.water:F0}";
        retardantText.text = $"Retardant: {resources.retardant:F0}";
        airSupportText.text = $"Air: {resources.air_support:F0}";
    }
}