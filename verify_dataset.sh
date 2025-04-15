# add run argument
# LIBRARY	VERSION	NAME_OF_CLASS_OR_FUNC	TYPE_OF_CHANGE	PROBLEM	STARTING_CODE	SOLUTION	TEST
library=$1 # maybe this is a list incase we need multiple libraries
version=$2
name_of_class_of_func=$3
type_of_change=$4
problem=$5 # "Write a function that renders the quadratic formula in LaTeX using Gradio's Chatbot using Gradio == 3.30.0. The quadratic formula is given by: x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}
python_code=$6
                    # assert render_quadratic_formula().beginswith("$") and render_quadratic_formula().endswith("$") 
additional_dep=$7
python_exec=$8
# Create a virtual environment
$python_exec -m venv --clear venv
# Activate the virtual environment
# source venv/bin/activate
venv/bin/pip install --upgrade pip==24.0
venv/bin/pip install $library==$version
if [ "$additional_dep" != "-" ]; then
    venv/bin/pip install $additional_dep
fi
venv/bin/pip freeze

timeout 60 venv/bin/python -c "$python_code"
 
# Capture the exit code
exit_code=$?
# Print the exit code
echo "THIS WAS THE EXIT CODE: $exit_code"
echo "$python_code"

# # Deactivate the virtual environment
# deactivate

# Remove the virtual environment directory
rm -rf venv


#from huggingface_hub import InferenceClient, AsyncInferenceClient