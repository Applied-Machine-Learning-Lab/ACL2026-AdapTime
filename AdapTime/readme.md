All experiments were conducted using two NVIDIA V100 GPUs or via DeepSeek-V3 API calls . 

The following Python libraries and packages were employed in our implementation:
torch==2.1.0
transformers==4.34.1
datasets==2.14.6
accelerate==0.33.0
trl==0.9.6
peft==0.12.0
openai==1.38.0


For running:

python Inference_in_context_learning.py --dataset TempReason_l3 --model deepseek-v3 --ICL --CoT

For Evaluation:

python Evaluation.py --dataset TempReason_l3 --model deepseek-v3 --ICL --CoT --type AdapTime_