OLD_REACT_EXAMPLE = """##Task
Your task is to use chemistry to create green paint. When you are done, focus on the green paint.
This room is called the bedroom. In it, you see: \n\tthe agent\n\ta substance called air\n\ta bed. On the bed is: a mattress. On the mattress is: a white pillow..\n\ta closet. The closet door is open. In the closet is: nothing. \n\ta picture\n\ta table. On the table is: nothing.\nYou also see:\n\tA door to the hallway (that is open)
##Interaction trajectory
[Action] think: To solve the task, I need to find materials to create green paint, use those materials, and focus on the object specified in the task once it is identified. The materials for making green paint are likely to be found in an art studio.
[Observation] OK.
[Action] teleport to art studio
[Observation] You teleport to the art studio.
[Action] look around
[Observation] This room is called the art studio. In it, you see: \n\tthe agent\n\ta substance called air\n\ta large cupboard. The large cupboard door is open. In the large cupboard is: a drawer\n\ta wood cup (containing yellow paint)\n\ta wood cup (containing blue paint)\n\ta wood cup (containing red paint)\n\ta table. On the table is: a glass cup (containing nothing).\n\ta wood cup (containing yellow paint)\n\ta wood cup (containing blue paint)\n\twood cup (containing red paint).\nYou also see:\n\tA door to the hallway (that is open)
[Action] think: I see blue paint here, which is one of the primary colors needed to mix green paint. I should start by pouring it into an empty container to mix with another color.
[Observation] OK.
[Action] pour cup containing blue paint in art studio in cup containing nothing
[Observation] You pour the contents of the wood cup into the glass cup.
[Action] pour cup containing yellow paint in art studio in cup containing blue paint in table
[Observation] You pour the contents of the wood cup into the glass cup.
[Action] think: Now that I have both blue and yellow paint in the same container, I need to mix them together to create green paint.
[Observation] OK.
[Action] mix cup containing blue paint and yellow paint
[Observation] Blue and yellow paint mix to produce green paint.
[Action] focus on green paint
"""

REACT_EXAMPLE = """##Task
Your task is to find a(n) plant. First, focus on the thing. Then, move it to the yellow box in the bedroom.\nThis room is called the kitchen. In it, you see: \n\tthe agent\n\ta substance called air\n\ta chair. On the chair is: nothing.\n\ta counter. On the counter is: a bowl (containing a red apple, a banana, an orange, a potato), a drawer.\n\ta cupboard. The cupboard door is closed. \n\ta freezer. The freezer door is closed. \n\ta fridge. The fridge door is closed. \n\ta glass jar (containing a substance called sodium chloride)\n\ta lighter\n\ta oven, which is turned off. The oven door is closed. \n\ta painting\n\ta sink, which is turned off. In the sink is: nothing.\n\ta substance called soap\n\ta stopwatch, which is deactivated. \n\ta stove, which is turned off. On the stove is: nothing.\n\ta table. On the table is: a glass cup (containing nothing).\n\ta thermometer, currently reading a temperature of 10 degrees celsius\nYou also see:\n\tA door to the bathroom (that is open)\n\tA door to the hallway (that is open)\n\tA door to the outside (that is open)\nIn your inventory, you see:\n\tan orange\n
##Interaction trajectory
[Action] think: I need to find a plant and move it to the yellow box in the bedroom. I should start by looking in the greenhouse.
[Observation] OK.
[Action] teleport to greenhouse
[Observation] You teleport to the greenhouse.
[Action] look around
[Observation] This room is called the greenhouse. In it, you see: 
        the agent
        a substance called air
        a bee hive. The bee hive door is closed. 
        a flower pot 4 (containing a substance called water)
        a flower pot 5 (containing a cherry tree in the adult stage, soil)
        a flower pot 6 (containing a lemon tree in the adult stage, soil)
        a flower pot 7 (containing a banana tree in the adult stage, soil)
        a flower pot 9 (containing a substance called water)
        a jug (containing nothing)
        a sink, which is turned off. In the sink is: nothing.
You also see:
        A door to the hallway (that is open)
        A door to the outside (that is open)
[Action] think: I see several plants in the greenhouse. I should pick one up and then teleport to the bedroom to move it to the yellow box. I will pick up the cherry tree.
[Observation] OK.
[Action] pick up cherry tree in the adult stage
[Observation] You move the cherry tree to the inventory.
[Action] teleport to bedroom
[Observation] You teleport to the bedroom.
[Action] look around
[Observation] This room is called the bedroom. In it, you see: \n\tthe agent\n\ta substance called air\n\ta bed. On the bed is: a mattress. On the mattress is: a white pillow..\n\ta closet. The closet door is closed. \n\ta picture\n\ta table. On the table is: nothing.\n\ta yellow box (containing nothing)\nYou also see:\n\tA door to the hallway (that is open)
[Action] move cherry tree in the adult stage to yellow box
[Observation] You move the cherry tree to the yellow box.
[Action] focus on cherry tree in the adult stage
[Observation] You focus on the cherry tree.
"""



CASE_PROMPT = """##Task
{task}
##Interaction trajectory
{trajectory}
"""

PROMPT = """You are a helpful assistant to do some scientific experiment in an environment. In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway. You should explore the environment and find the items you need to complete the experiment. You can teleport to any room in one step. All containers in the environment have already been opened, you can directly get items from the containers.

Here are some relevant cases:
{case_prompt}

Here is the task description, and the initial observation by taking action look around:
{task}
Here are all the action templates and their corresponding explanation.
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step
Beyond these actions, you may use "think:" action to perform basic reasoning, but only when it is necessary.
Now it’s your turn to take actions. Please output the action using the action template above. Let's start!
"""

SUFFIX_PROMPT = """
Please directly output the action with the format: [Action] action
If you are required to enter a number, please output the action with the format: [Action] number
"""

HISTORY_PROMPT = """[Action] {action}
[Observation] {state}
"""


