import json
import argparse
import sys

def transform_entry(entry):
    """
    Transforms a single dictionary entry from instruction/output
    to problem/solution keys.
    """
    # We use .get() to avoid errors if keys are missing, 
    # though strict enforcement can be added if needed.
    return {
        "problem": entry.get("instruction", ""),
        "solution": entry.get("output", "")
    }

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert JSON data from 'instruction/output' format to 'problem/solution' format."
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to save the converted JSON file")
    
    args = parser.parse_args()

    try:
        # Open and read the input file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if data is a list (multiple records) or a dict (single record)
        if isinstance(data, list):
            print(f"Detected list of {len(data)} items. Converting...")
            new_data = [transform_entry(item) for item in data]
        elif isinstance(data, dict):
            print("Detected single JSON object. Converting...")
            new_data = transform_entry(data)
        else:
            print("Error: The root of the JSON file must be a list or an object.")
            sys.exit(1)

        # Write the transformed data to the output file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully saved converted data to '{args.output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: '{args.input_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()