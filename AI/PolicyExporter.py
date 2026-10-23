import torch
import torch.nn as nn
from typing import Tuple
from AI.OptionCriticNetwork import OptionCriticNetwork

class PolicyExporter:
    """Export optimized policy for XR deployment."""
    @staticmethod
    def export_to_onnx(model: nn.Module, input_shape: Tuple[int, ...], output_path: str):
        """
        Exports the OptionCriticNetwork model to ONNX format.
        Handles multiple output heads.

        Args:
            model (nn.Module): The OptionCriticNetwork instance.
            input_shape (Tuple[int, ...]): Expected input shape (channels, height, width).
            output_path (str): Path to save the ONNX model.
        """
        model.eval()
        dummy_input = torch.randn(1, *input_shape)  # Shape: (1, channels, height, width)
        
        output_names = [
            'option_type_logits',
            'target_xy_logits',
            'termination_logit',
            'value_output'
        ]

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=output_names,
            dynamic_axes={'observation': {0: 'batch_size'}}
        )
        print(f"Model exported to ONNX at: {output_path}")

    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        """Dynamically quantizes the model for reduced size and faster inference."""
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model.cpu(), {nn.Linear, nn.Conv2d}, dtype=torch.qint8 # Quantize both Linear and Conv2d
        )
        print("Model dynamically quantized.")
        return quantized_model

    @staticmethod
    def extract_policy_inference_model(original_model: OptionCriticNetwork) -> nn.Module:
        """
        Extracts a simplified model for policy inference, excluding the value head.
        This model will output option type, x, y logits, and termination logit.
        """
        class PolicyInferenceOnly(nn.Module):
            def __init__(self, original_model_ref: OptionCriticNetwork):
                super().__init__()
                self.conv_layers = original_model_ref.conv_layers
                self.fc_layers = original_model_ref.fc_layers
                self.option_type_head = original_model_ref.option_type_head
                self.target_x_head = original_model_ref.target_x_head
                self.target_y_head = original_model_ref.target_y_head
                self.termination_head = original_model_ref.termination_head

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                conv_out = self.conv_layers(x)
                flat_out = conv_out.view(conv_out.size(0), -1)
                features = self.fc_layers(flat_out)
                
                option_type_logits = self.option_type_head(features)
                target_x_logits = self.target_x_head(features)
                target_y_logits = self.target_y_head(features)
                termination_logit = self.termination_head(features)
                
                return option_type_logits, target_x_logits, target_y_logits, termination_logit
        
        policy_inference_model = PolicyInferenceOnly(original_model)
        policy_inference_model.eval()
        print("Policy inference model extracted (excluding value head).")
        return policy_inference_model