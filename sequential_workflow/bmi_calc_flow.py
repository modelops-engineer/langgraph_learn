from langgraph.graph import StateGraph,START, END   # START, END Dummy node
from typing import TypedDict


class BMIState(TypedDict):
    weight : float
    height : float
    category : str
    bmi : float

# define graph
graph = StateGraph(BMIState)

def bmi_calc(state : BMIState) -> BMIState:
    weight = state['weight']
    height = state['height']
    
    bmi = weight/(height**2)

    state['bmi'] = round(bmi,2)

    return state    


def bmi_label(state: BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = "Underweight"
    elif bmi < 25:
        state['category'] = "Normal"
    elif bmi < 30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obese"

    return state

# add nodes
node1 = graph.add_node('calculate_bmi', bmi_calc)
node2 = graph.add_node('label_bmi', bmi_label)

# add edges
edge1 = graph.add_edge(START, 'calculate_bmi')
edge2 = graph.add_edge('calculate_bmi', 'label_bmi')
edge3 = graph.add_edge('label_bmi', END)

# compile
workflow = graph.compile()

# execute
initial_state = {
    'weight':78,
    'height':1.73
}
output_state = workflow.invoke(initial_state)
print(output_state)

# png_bytes = workflow.get_graph().draw_mermaid_png()
# with open("workflow_graph.png", "wb") as f:
#     f.write(png_bytes)