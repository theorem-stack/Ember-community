// SimulationDataLoader.cs

using UnityEngine;
using System.IO;
using System.Collections.Concurrent;
using System.Threading;
using Newtonsoft.Json;

public class SimulationDataLoader : MonoBehaviour
{
    public event System.Action<EpisodeStepData> OnNewEpisodeStepDataAvailable;
    public event System.Action<InitialEnvironmentData> OnInitialEnvironmentDataAvailable;

    private readonly ConcurrentQueue<EpisodeStepData> episodeDataQueue = new ConcurrentQueue<EpisodeStepData>();
    private readonly ConcurrentQueue<InitialEnvironmentData> initialDataQueue = new ConcurrentQueue<InitialEnvironmentData>();

    public EpisodeStepData CurrentEpisodeStepData { get; private set; }
    public InitialEnvironmentData CurrentInitialEnvironmentData { get; private set; }

    // Start a background thread to handle file loading for initial data.
    public void LoadInitialEnvironmentData(string filePath)
    {
        ThreadPool.QueueUserWorkItem(state =>
        {
            if (File.Exists(filePath))
            {
                try
                {
                    string jsonString = File.ReadAllText(filePath);
                    InitialEnvironmentData data = JsonConvert.DeserializeObject<InitialEnvironmentData>(jsonString);
                    initialDataQueue.Enqueue(data);
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Error loading initial environment data on thread: {e.Message}");
                }
            }
            else
            {
                Debug.LogError($"Initial environment data file not found on thread at: {filePath}");
            }
        });
    }

    // Start a background thread to handle file loading for episode data.
    public void StartAsyncLoad(string filePath)
    {
        ThreadPool.QueueUserWorkItem(state =>
        {
            if (File.Exists(filePath))
            {
                try
                {
                    string jsonString = File.ReadAllText(filePath);
                    EpisodeStepData data = JsonConvert.DeserializeObject<EpisodeStepData>(jsonString);
                    episodeDataQueue.Enqueue(data);
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Error loading episode visualization data on thread: {e.Message}");
                }
            }
            else
            {
                Debug.LogError($"Episode visualization data file not found on thread at: {filePath}");
            }
        });
    }

    void Update()
    {
        // Process any new initial data on the main thread
        while (initialDataQueue.TryDequeue(out InitialEnvironmentData initialData))
        {
            CurrentInitialEnvironmentData = initialData;
            OnInitialEnvironmentDataAvailable?.Invoke(initialData);
            Debug.Log("Successfully loaded initial environment data on main thread.");
        }

        // Process any new episode data on the main thread
        while (episodeDataQueue.TryDequeue(out EpisodeStepData episodeData))
        {
            CurrentEpisodeStepData = episodeData;
            OnNewEpisodeStepDataAvailable?.Invoke(episodeData);
            Debug.Log("Successfully loaded new episode step data on main thread.");
        }
    }
}