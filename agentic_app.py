import streamlit as st
import sys
from io import StringIO
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.language_models import BaseLLM
from typing import Optional, List, Mapping, Any

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

class MockLocalLLM(BaseLLM):
    @property
    def _llm_type(self) -> str:
        return "mock_local_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if "power factor" in prompt.lower() or "calculate" in prompt.lower():
            return "Thought: I need to use the power_system_calculator tool.\nAction: power_system_calculator\nAction Input: 480 * 100 * 0.85 * (3**0.5) / 1000"
        elif "grounding resistance" in prompt.lower() or "standard" in prompt.lower():
            return "Thought: I need to look up the standard.\nAction: retrieve_standard_document\nAction Input: grounding resistance"
        else:
            return "Thought: No tool needed.\nFinal Answer: The general principles of Agentic AI involve planning, tool use, and reflection."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "MockLocalLLM"}

@st.cache_resource
def get_agent(model_key: str):
    model_id = LOCAL_MODELS[model_key]
    st.subheader("Agent Setup Status")
    st.info(f"Selected Model: **{model_key}** (HF ID: `{model_id}`)")
    llm = MockLocalLLM()
    st.success("Mock Local LLM is active for demonstration purposes.")
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    st.success("Agentic Engineer Framework Initialized (ReAct - LangChain 0.2+).")
    return agent_executor

def capture_agent_output(agent_executor, goal):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        response = agent_executor.invoke({"input": goal})
        verbose_output = sys.stdout.getvalue()
        final_answer = response["output"]
        return verbose_output, final_answer
    finally:
        sys.stdout = old_stdout

def main():
    st.set_page_config(page_title="Agentic AI Engineer Demo", layout="wide")
    st.title("Agentic AI System for Power Engineering Demo")
    st.caption("Illustrating Agentic AI: Planning (Thought), Action (Tool Use), and Reflection.")
    st.markdown("---")

    with st.sidebar:
        st.header("Model Configuration")
        selected_model = st.selectbox(
            "Select a Local LLM Model (No API Key)",
            options=MODEL_IDS,
            index=0
        )

    agent_executor = get_agent(selected_model)
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
        with st.spinner(f"Agentic System (using {selected_model}) is thinking..."):
            verbose_output, final_answer = capture_agent_output(agent_executor, prompt)

        st.markdown("---")
        st.header("2. Agentic Workflow (ReAct Steps)")
        st.code(verbose_output, language="text")

        st.markdown("---")
        st.header("3. Final Answer")
        st.balloons()
        st.success(final_answer)

if __name__ == "__main__":
    main()
