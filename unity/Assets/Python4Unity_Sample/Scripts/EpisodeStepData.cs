using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Main container for all visualization data
[System.Serializable]
public class EpisodeStepData
{
    public List<FireData> fires;
    public List<EmberData> embers;
    public List<PlumeData> plumes;
    // public WindFieldData wind_field;
    public List<AgentData> agents;
    public int step_count;
    public float time_elapsed;
    public List<float[]> agent_obs_heat_flux;
    public GridMetadata grid_metadata;
}

// Data for an individual fire
[System.Serializable]
public class FireData
{
    public float[] position; // A list of 3 floats (x, y, z) where z-axis is vertical in my simulation. Should flip z and y in Unity.
    public float intensity;
    public float temperature;
    public float fuel_remaining;
    public float fire_age;
}

// Data for an individual ember
[System.Serializable]
public class EmberData
{
    public float[] position; // A list of 3 floats (x, y, z) where z-axis is vertical in my simulation. Should flip z and y in Unity.
    public float density;
}

// Data for plume dynamics
[System.Serializable]
public class PlumeData
{
    public float[] origin_position; // The new origin for the plume (x, y, z) where y should be switched with z-axis in Unity.
    public float plume_u;   // Mean horizontal wind X
    public float plume_v;   // Mean horizontal wind Y
    public float plume_w;   // Combined vertical velocity (buoyancy + ambient vertical wind)
    public float total_heat; // For overall plume size/intensity
    public float h_sub; // Plume height
    public float ell;   // Plume length/tilt
    public float k; // Turbulence factor
}

// Data for the wind field
// [System.Serializable]
// public class WindFieldData
// {
//     public List<float> u;
//     public List<float> v;
//     public List<float> w;
//     public int[] shape; // A list of 3 ints (for a 3D grid)
// }

// Data for an individual agent
[System.Serializable]
public class AgentData
{
    public int agent_id;
    public AICoordData _ai;
    public SimCoordData _sim;
    public IntentData intent;
    public ResourcesData resources;
}

[System.Serializable]
public class AICoordData
{
    public float[] position; // [agent_obs_x, agent_obs_y]
    public List<float[]> path; // List of [agent_obs_x, agent_obs_y]
}

[System.Serializable]
public class SimCoordData
{
    public float[] position; // [sim_x_start, sim_y_start]
    public List<float[]> path; // List of [sim_x_start, sim_y_start]
}

// Data for an agent's intent
[System.Serializable]
public class IntentData
{
    public float x;
    public float y;
    public int type;
    public float sim_x_start;
    public float sim_x_end;
    public float sim_y_start;
    public float sim_y_end;
    public bool agent_applied_action;
}

// Data for an agent's resources
[System.Serializable]
public class ResourcesData
{
    public float water;
    public float retardant;
    public float air_support;
}

// Metadata for the simulation grid
[System.Serializable]
public class GridMetadata
{
    public int nx;
    public int ny;
    public float dx;
    public float dy;
    public List<float> z_levels;
    public List<float> dz;
    public int agent_obs_x;
    public int agent_obs_y;
}