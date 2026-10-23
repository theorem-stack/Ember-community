Shader "Custom/UIGridShader"
{
    Properties
    {
        _HeatFluxTex("Heat Flux Texture", 2D) = "white" {}
        _BurnBoxTex("Burn Box Texture", 2D) = "white" {} // New: Burn box mask
        _GridSize("Grid Size", Vector) = (10, 10, 0, 0)
        _GridDotRadius("Grid Dot Radius", Float) = 0.005
        _AgentRadius("Agent Radius", Float) = 0.05
        _TargetRadius("Target Radius", Float) = 0.05
        _PathDotRadius("Path Dot Radius", Float) = 0.005
        _BackgroundColor("Background Color", Color) = (0.1, 0.15, 0.2, 0.5)
        _GlowIntensity("Glow Intensity", Float) = 1.5
        _PulseSpeed("Pulse Speed", Float) = 2.0
        _HeatGradientStart("Heat Gradient Start", Color) = (0, 0.5, 1, 0.3)
        _HeatGradientEnd("Heat Gradient End", Color) = (1, 0, 0, 0.8)
        _BurnBoxColor("Burn Box Color", Color) = (1, 0.2, 0, 0.2) // New: Burn box tint
    }
    SubShader
    {
        Tags { "Queue"="Overlay" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _HeatFluxTex;
            sampler2D _BurnBoxTex; // New: Burn box texture
            float4 _GridSize;
            float _GridDotRadius;
            float _AgentRadius;
            float _TargetRadius;
            float _PathDotRadius;
            int _AgentCount;
            int _TargetCount;
            int _PathCount;
            fixed4 _BackgroundColor;
            float _GlowIntensity;
            float _PulseSpeed;
            fixed4 _HeatGradientStart;
            fixed4 _HeatGradientEnd;
            fixed4 _BurnBoxColor; // New: Burn box color

            StructuredBuffer<float4> _AgentData;
            StructuredBuffer<float4> _TargetData;
            StructuredBuffer<float4> _PathData;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float RenderFilledCircle(float2 uv, float2 pos, float radius)
            {
                float dist = distance(uv, pos);
                return smoothstep(radius, radius * 0.5, dist) * _GlowIntensity;
            }

            float RenderOutlinedCircle(float2 uv, float2 pos, float radius, float thickness)
            {
                float dist = distance(uv, pos);
                float inner = radius - thickness;
                float outer = radius;
                float innerEdge = smoothstep(inner, inner * 0.8, dist);
                float outerEdge = smoothstep(outer, outer * 0.8, dist);
                float2 dir = uv - pos;
                float angle = atan2(dir.y, dir.x);
                float crosshair = abs(sin(angle * 4.0)) * 0.5 + 0.5;
                return (innerEdge - outerEdge) * _GlowIntensity * crosshair;
            }

            float Noise(float2 uv)
            {
                return frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col = _BackgroundColor;
                float noise = Noise(i.uv * 10.0) * 0.05;
                col.rgb += noise;

                // Apply burn box tint
                float burnBox = tex2D(_BurnBoxTex, i.uv).r;
                col = lerp(col, _BurnBoxColor, burnBox * _BurnBoxColor.a);

                float heat = tex2D(_HeatFluxTex, i.uv).r;
                fixed4 heatColor = lerp(_HeatGradientStart, _HeatGradientEnd, heat);
                col = lerp(col, heatColor, heatColor.a * heat);

                float2 gridUV = i.uv * _GridSize.xy;
                float2 gridCell = floor(gridUV);
                float2 nearestGridPoint = gridCell + 0.5;
                float2 uvToGridPoint = i.uv - (nearestGridPoint / _GridSize.xy);
                float gridDot = RenderFilledCircle(i.uv, nearestGridPoint / _GridSize.xy, _GridDotRadius);
                float flicker = (sin(_Time.y * _PulseSpeed + Noise(nearestGridPoint)) * 0.5 + 0.5) * 0.2 + 0.8;
                col.rgb = lerp(col.rgb, fixed3(0.5, 0.8, 1.0), gridDot * flicker * 0.5);

                float pulse = (sin(_Time.y * _PulseSpeed) * 0.5 + 0.5) * 0.3 + 0.7;

                for (int idx = 0; idx < _PathCount * 2; idx += 2)
                {
                    float4 pathPosColor = _PathData[idx];
                    float4 pathColorB = _PathData[idx + 1];
                    float2 pathPos = pathPosColor.xy;
                    if (pathPos.x >= 0 && pathPos.y >= 0)
                    {
                        float pathDot = RenderFilledCircle(i.uv, pathPos, _PathDotRadius);
                        fixed3 pathColor = fixed3(pathPosColor.z, pathPosColor.w, pathColorB.z);
                        col.rgb = lerp(col.rgb, pathColor * pulse, pathDot);
                    }
                }

                for (int idx = 0; idx < _AgentCount * 2; idx += 2)
                {
                    float4 agentPosColor = _AgentData[idx];
                    float4 agentColorB = _AgentData[idx + 1];
                    float2 agentPos = agentPosColor.xy;
                    if (agentPos.x >= 0 && agentPos.y >= 0)
                    {
                        float agent = RenderFilledCircle(i.uv, agentPos, _AgentRadius);
                        fixed3 agentColor = fixed3(agentPosColor.z, agentPosColor.w, agentColorB.z);
                        col.rgb = lerp(col.rgb, agentColor * pulse, agent);
                    }
                }

                for (int idx = 0; idx < _TargetCount * 2; idx += 2)
                {
                    float4 targetPosColor = _TargetData[idx];
                    float4 targetColorB = _TargetData[idx + 1];
                    float2 targetPos = targetPosColor.xy;
                    if (targetPos.x >= 0 && targetPos.y >= 0)
                    {
                        float target = RenderOutlinedCircle(i.uv, targetPos, _TargetRadius, _TargetRadius * 0.2);
                        fixed3 targetColor = fixed3(targetPosColor.z, targetPosColor.w, targetColorB.z);
                        col.rgb = lerp(col.rgb, targetColor * pulse, target);
                    }
                }

                return col;
            }
            ENDCG
        }
    }
}