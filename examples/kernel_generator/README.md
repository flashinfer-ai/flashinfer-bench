# Kernel Generator

A multi-turn kernel generating agent that uses FlashInfer-Bench for evaluation feedback. It can conduct sequential multi-turn generation and beam search kernel exploration.

## Usage

1. **Configure generation settings** in `kernel_generator_example.py`:
   - Set `model_name` (e.g., `"gpt-5-2025-08-07"`)
   - Set `language` (`"cuda"` or `"triton"`, will support more in the future)
   - Set `target_gpu` (e.g., `"B200"`, `"H100"`, `"A100"`)
   - Optionally set `definition` to target a specific kernel (leave empty to generate all definitions in the traceset)

2. **Set traceset path**:
   - Update `traceset_path` to your flashinfer-trace dataset directory

3. **To Enable beam search**:
   - Uncomment lines 97-98 to use beam search mode

4. **Set API credentials**:
   - Create a `.env` file by following the .env.example:
     ```
     LLM_API_KEY=your_api_key
     BASE_URL=your_base_url  # Optional, for non-OpenAI APIs
     ```

5. **Run the generator**:
   ```bash
   python kernel_generator_example.py
   ```

Generated solutions are saved to `{traceset_path}/solutions/{author}/{op_type}/{definition_name}/{solution_name}.json`
