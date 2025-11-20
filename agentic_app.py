import streamlit as st
import sys
from io import StringIO
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

# --- IMPORTS FOR REAL LOCAL MODELS ---
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# --------------------------------------

LOCAL_MODELS = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2b-it": "google/gemma-2b-it",
    "Zephyr-7B-beta": "HuggingFaceH4/zephyr-7b-beta",
}
MODEL_IDS = list(LOCAL_MODELS.keys())

@tool
def power_system_calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

@tool
def retrieve_standard_document(topic: str) -> str:
    if "grounding resistance" in topic.lower():
        return "The typical maximum acceptable grounding resistance for a major substation is 5 Ohms, as per IEEE 80 standard guidelines."
    elif "feeder voltage level" in topic.lower():
        return "Standard distribution feeder voltages in the US are commonly 12.47 kV and 34.5 kV."
    else:
        return f"No specific standard found for the topic: {topic}. Try a different query."

tools = [power_system_calculator, retrieve_standard_document]

class MockLocalLLM(tool.BaseLLM):
    @property
    def _llm_type(self) -> str:
        return "mock_local_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if "power factor" in prompt.lower() or "calculate" in prompt.lower():
            return "Thought: I need to use the power_system_calculator tool to solve this math problem. Action: power_system_calculator[480 * 100 * 0.85 * (3**0.5) / 1000]"
        elif "grounding resistance" in prompt.lower() or "standard" in prompt.lower():
            return "Thought: I need to look up an electrical standard. Action: retrieve_standard_document[grounding resistance]"
        else:
            return "Thought: I don't need a specific tool. Final Answer: The general principles of Agentic AI involve planning, tool use, and reflection."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "MockLocalLLM"}

@st.cache_resource
def get_agent(model_key: str):
    model_id = LOCAL_MODELS[model_key]
    st.subheader("Agent Setup Status")
    st.info(f"Selected Model: **{model_key}** (HF ID: `{model_id}`)")

    # --- REAL LLM LOADING (UNCOMMENT FOR ACTUAL USE) ---
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,
    #         load_in_8bit=True,
    #     )
    #     pipe = pipeline(
    #         "text-generation",
    #         model=model,
    #         tokenizer=tokenizer,
    #         max_new_tokens=256,
    #         temperature=0.1,
    #         trust_remote_code=True,
    #         device_map="auto"
    #     )
    #     llm = HuggingFacePipeline(pipeline=pipe)
    #     st.success("✅ Real Local Model and Pipeline Loaded successfully.")
    # except Exception as e:
    #     st.error(f"❌ Failed to load REAL model: {e}. Falling back to Mock LLM.")
    # ---------------------------------------------------
    
    llm = MockLocalLLM()
    st.success("✅ Mock Local LLM is active for demonstration purposes.")

    agentic_engineer = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    st.success("✅ Agentic Engineer Framework Initialized (ReAct).")
    return agentic_engineer

def capture_agent_output(agent, goal):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        result = agent.run(goal)
        verbose_output = sys.stdout.getvalue()
        return verbose_output, result
    finally:
        sys.stdout = old_stdout

def main():
    st.set_page_config(page_title="Agentic AI Engineer Demo", layout="wide")
    st.title("⚡ Agentic AI System for Power Engineering Demo")
    st.caption("Illustrating Agentic AI: Planning (Thought), Action (Tool Use), and Reflection.")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Model Configuration")
        selected_model = st.selectbox(
            "Select a Local LLM Model (No API Key)",
            options=MODEL_IDS,
            index=0,
            help="This selects the 'Brain' of the Agentic System. Only the Mock LLM is active unless you uncomment the real loading code."
        )
        st.markdown("""
        ---
        ### ⚙️ To Use a REAL Model:
        1.  Install dependencies: `pip install langchain transformers accelerate bitsandbytes torch`
        2.  **Uncomment** the "REAL LLM LOADING" section in the code.
        3.  Ensure your machine has sufficient VRAM (8GB+ recommended).
        """)

    agentic_engineer = get_agent(selected_model)
    st.markdown("---")

    st.header("1. Input Goal")
    prompt = st.text_input(
        "Enter a complex engineering goal for the Agent to solve:",
        value="A 480V three-phase motor draws 100 Amps with a power factor of 0.85. Calculate the total power in kW.",
        placeholder="e.g., What is the maximum grounding resistance for a major substation?"
    )

    if st.button("Run Agentic System"):
        if not prompt:
            st.error("Please enter a goal.")
            return

        with st.spinner(f"Agentic System (using {selected_model}) is planning and executing..."):
            verbose_output, final_answer = capture_agent_output(agentic_engineer, prompt)

        st.markdown("---")
        st.header("2. Agentic Workflow (ReAct Steps)")
        st.info("The Agent's internal decision-making process (Thought, Action, Observation) is displayed below, demonstrating autonomy.")
        
        st.code(verbose_output, language='text')
        
        st.markdown("---")
        st.header("3. Final Answer")
        st.balloons()
        st.success(final_answer)

if __name__ == "__main__":
    main()
