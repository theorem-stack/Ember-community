using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;
using System.Collections;

public class UIMissionLogger : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI logPrefab;
    [SerializeField] private ScrollRect scrollRect;
    [SerializeField] private Image panelBackground;
    [SerializeField] private Color warningTextColor = new Color(1f, 0.6f, 0f, 1f); // Orange for warnings
    [SerializeField] private float pulseSpeed = 2.0f;
    [SerializeField] private int maxLogEntries = 50;

    // Dictionary to map integer intent types to text strings
    private Dictionary<int, string> actionTypeMap = new Dictionary<int, string>
    {
        {0, "water"},
        {1, "firebreak"},
        {2, "retardant"},
        {3, "air_support"},
        {4, "idle"}
    };

    private List<TextMeshProUGUI> logEntries = new List<TextMeshProUGUI>();
    private Queue<TextMeshProUGUI> logPool = new Queue<TextMeshProUGUI>();

    private void Awake()
    {
        if (scrollRect == null)
        {
            scrollRect = GetComponent<ScrollRect>();
        }
        if (scrollRect == null)
        {
            Debug.LogError("ScrollRect component is missing on the UIMissionLogger.");
            enabled = false;
        }
    }

    private void Start()
    {
        if (panelBackground != null)
        {
            panelBackground.color = new Color(0.1f, 0.15f, 0.2f, 0.3f);
        }
        InitializeObjectPool();
        UpdateContentWidth();
        LayoutRebuilder.ForceRebuildLayoutImmediate(scrollRect.content);
    }

    private void Update()
    {
        if (panelBackground != null)
        {
            float pulse = (Mathf.Sin(Time.time * pulseSpeed) * 0.5f + 0.5f) * 0.2f + 0.8f;
            panelBackground.color = new Color(panelBackground.color.r, panelBackground.color.g, panelBackground.color.b, pulse * 0.5f);
        }
    }

    private void InitializeObjectPool()
    {
        for (int i = 0; i < maxLogEntries; i++)
        {
            TextMeshProUGUI newLog = Instantiate(logPrefab, scrollRect.content);
            newLog.gameObject.SetActive(false);
            newLog.alignment = TextAlignmentOptions.Left; // Ensure horizontal text
            newLog.rectTransform.localRotation = Quaternion.identity; // Reset rotation
            newLog.rectTransform.sizeDelta = new Vector2(scrollRect.content.rect.width - 20, 30);
            newLog.rectTransform.anchorMin = new Vector2(0, 1);
            newLog.rectTransform.anchorMax = new Vector2(0, 1);
            newLog.rectTransform.pivot = new Vector2(0, 1);
            newLog.rectTransform.anchoredPosition = Vector2.zero;
            logPool.Enqueue(newLog);
        }
    }

    private void UpdateContentWidth()
    {
        RectTransform viewportRect = scrollRect.viewport;
        RectTransform contentRect = scrollRect.content;
        float viewportWidth = viewportRect.rect.width;
        contentRect.sizeDelta = new Vector2(viewportWidth - 20, contentRect.sizeDelta.y); // 20 for scrollbar
    }

    public void LogAgentUpdates(List<AgentData> agents)
    {
        if (agents == null || agents.Count == 0)
        {
            string logLine = $"[{System.DateTime.Now:HH:mm:ss}]".PadRight(10) + "No Agents Available".PadRight(70);
            AppendLogEntry(logLine, warningTextColor, -1);
            return;
        }

        foreach (var agent in agents)
        {
            string intentText = "N/A";

            if (agent.intent != null)
            {
                if (actionTypeMap.TryGetValue(agent.intent.type, out string typeText))
                {
                    intentText = typeText;
                }
                else
                {
                    intentText = $"Unknown type ({agent.intent.type})";
                }
            }

            string timestamp = $" [{System.DateTime.Now:HH:mm:ss}]".PadRight(21); // 10 chars
            string agentIdText = $"Agent {agent.agent_id}".PadRight(16); // 50 chars
            string intentTextPadded = intentText.PadRight(20); // 20 chars
            string logLine = $"{timestamp}{agentIdText}{intentTextPadded}";
            AppendLogEntry(logLine, GenerateColorFromAgentId(agent.agent_id), agent.agent_id);
        }
    }

    public void AppendLogEntry(string logText, Color textColor, int agentId = -1)
    {
        TextMeshProUGUI logEntry;

        if (logPool.Count > 0)
        {
            logEntry = logPool.Dequeue();
            logEntry.gameObject.SetActive(true);
        }
        else
        {
            logEntry = logEntries[0];
            logEntries.RemoveAt(0);
        }

        logEntry.text = logText;
        logEntry.color = textColor;
        logEntry.alignment = TextAlignmentOptions.Left;
        logEntry.rectTransform.localRotation = Quaternion.identity;
        float contentWidth = scrollRect.content.rect.width - 20;
        logEntry.rectTransform.sizeDelta = new Vector2(contentWidth, 30);
        logEntry.rectTransform.anchorMin = new Vector2(0, 1);
        logEntry.rectTransform.anchorMax = new Vector2(0, 1);
        logEntry.rectTransform.pivot = new Vector2(0, 1);
        logEntry.rectTransform.anchoredPosition = Vector2.zero;
        logEntry.transform.SetAsLastSibling();

        logEntries.Add(logEntry);

        // Only start coroutines if the parent GameObject is active.
        if (gameObject.activeInHierarchy)
        {
            StartCoroutine(FadeInLogEntry(logEntry));
            StartCoroutine(SmoothScrollToBottom());
        }

        LayoutRebuilder.ForceRebuildLayoutImmediate(scrollRect.content);
    }

    private IEnumerator FadeInLogEntry(TextMeshProUGUI logEntry)
    {
        CanvasGroup canvasGroup = logEntry.GetComponent<CanvasGroup>();
        if (canvasGroup == null)
        {
            canvasGroup = logEntry.gameObject.AddComponent<CanvasGroup>();
        }

        canvasGroup.alpha = 0f;
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * 5f;
            canvasGroup.alpha = t;
            yield return null;
        }
    }

    private IEnumerator SmoothScrollToBottom()
    {
        float startNormalizedPosition = scrollRect.verticalNormalizedPosition;
        float endNormalizedPosition = 0f;
        float elapsedTime = 0f;
        float duration = 0.5f;

        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            scrollRect.verticalNormalizedPosition = Mathf.Lerp(startNormalizedPosition, endNormalizedPosition, elapsedTime / duration);
            yield return null;
        }
        scrollRect.verticalNormalizedPosition = endNormalizedPosition;
    }

    private Color GenerateColorFromAgentId(int agentId)
    {
        const float goldenRatioConjugate = 0.61803398875f;
        float hue = (agentId * goldenRatioConjugate) % 1.0f;
        float saturation = 0.8f;
        float value = 0.95f;
        return Color.HSVToRGB(hue, saturation, value);
    }
}