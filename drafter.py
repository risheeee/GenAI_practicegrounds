from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Updates the document with provided content"""
    global document_content
    document_content = content
    return f"Document has been succcessfully updated. current content: {document_content}"

@tool
def save(filename: str) -> str:
    """saves the current document to a text file
    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as f:
            f.write(document_content)
        print(f"Document has been saved to: {filename}")
        return f"Document has been saved to: {filename}"
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, save]

model = ChatOllama(model="llama3.1:latest", temperature=0).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = f"""
    You are a drafter, a helpful writing assistant. You are going to help the user to update and modfy the documents.
    - If the user wants to update or modify the contentt, use the 'update' tools with the complete updated content.
    - If the user wants to save and finish, use the 'save' tool.
    - Make ure to always show the current document state after modifications.
    The current document content is: {document_content}
    """)

    if not state['messages']:
        user_input = "I'm ready to help you update the document, what would you like to create?"
        user_message = HumanMessage(content = user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content = user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nusing tools: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": list(state['messages']) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state['messages']
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if(isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function to print messages  in a more readable format"""
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\ntool result: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    },
)
app = graph.compile()

def run_doc_agent():
    print("\n === DRAFTER ===")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step['messages'])
    print("\n === DRAFTER END ===")

if __name__ == "__main__":
    run_doc_agent()