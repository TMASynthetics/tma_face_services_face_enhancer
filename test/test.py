import os, sys, time, json

# Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Pipeline class from the src.pipeline module
from src.pipeline import Pipeline

# Record the start time of the execution
start_time = time.time()

# Initialize the Pipeline with input image and analyzer response file
pipeline = Pipeline('test/test_data/marie.jpeg', "test/test_data/analyser_response.json")

# Run the pipeline and get the output data
output_data = pipeline.run()

# Record the end time of the execution
end_time = time.time()

# Print the execution time
print(f"Execution time: {round(end_time - start_time, 2)} seconds")

# Write the output data to a JSON file
with open('test/test_data/output_data.json', 'w') as outfile:
	json.dump(output_data, outfile)

# Indicate that the job is done
print("job done")