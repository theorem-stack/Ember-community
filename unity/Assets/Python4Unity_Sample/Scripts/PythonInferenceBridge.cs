using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Python4Unity.Sample
{
    public class PythonInferenceBridge : MonoBehaviour
    {
        private IPyObject pythonModule;
        private IPython py; // Store the interpreter instance
        private const string PythonModuleName = "InferFirefighterOptionCritic";
        private const string setup_environment_fun = "setup_environment";
        private const string setup_agent_fun = "setup_agent";
        private const string setup_episode_fun = "setup_episode";
        private const string step_episode_fun = "step_episode";
        private const string get_status_fun = "get_status";

        // Pretrained Model Specs
        private const string model_filename = "TrainedModels/firefighter_policy_batch_0_20250804_143554.onnx";
        private const string config_filename = "TrainedModels/firefighter_policy_config_20250804_143517.json";

        // Reference to the new data loader script
        public SimulationDataLoader simulationDataLoader; // Assign in Inspector

        void Start()
        {
            // Ensure the DataLoader is assigned
            if (simulationDataLoader == null)
            {
                UnityEngine.Debug.LogError("SimulationDataLoader not assigned to PythonInferenceBridge!");
                return;
            }

            string environmentOutputFilePath = Path.Combine(Application.persistentDataPath, "environment_data.json");
            string episodeOutputFilePath = Path.Combine(Application.persistentDataPath, "episode_data.json");

            py = PythonUtil.GetInstance();
            pythonModule = py.GetModule(PythonModuleName);

            // Setup Environment
            var result1 = pythonModule.CallAttr<string>(setup_environment_fun, config_filename, environmentOutputFilePath);
            UnityEngine.Debug.Log(result1);

            // Load initial environment data using the dedicated loader
            simulationDataLoader.LoadInitialEnvironmentData(environmentOutputFilePath);

            // Setup Agent
            var result2 = pythonModule.CallAttr<string>(setup_agent_fun, model_filename, config_filename);
            UnityEngine.Debug.Log(result2);

            // Setup Episode
            var result3 = pythonModule.CallAttr<string>(setup_episode_fun, episodeOutputFilePath);
            UnityEngine.Debug.Log(result3);

            // Start the main simulation loop as a coroutine
            StartCoroutine(RunSimulationLoop(episodeOutputFilePath));
        }

        private IEnumerator RunSimulationLoop(string episodeOutputFilePath)
        {
            var done = false;
            for (int i = 0; i < 80; i++)
            {
                // Step Episode
                var result4 = pythonModule.CallAttr<string>(step_episode_fun);
                UnityEngine.Debug.Log($"Python Step Result: {i} " + result4);

                // Instead of a direct blocking call, trigger the async load
                simulationDataLoader.StartAsyncLoad(episodeOutputFilePath);

                done = pythonModule.CallAttr<bool>(get_status_fun);

                // Pause for 5 seconds before the next simulation step
                yield return new WaitForSeconds(5f);
            }
            UnityEngine.Debug.Log("Completed Simulation!!");
        }

        void OnDestroy()
        {
            // Dispose of the Python interpreter instance when the component is destroyed
            if (py != null)
            {
                py.Dispose();
            }
        }
    }
}