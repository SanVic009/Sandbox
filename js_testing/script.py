import sys
import json

def factorial(n):
    if n < 0:
        return None  # Indicate invalid input
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "No number provided"}))
            sys.exit(1)

        num = int(sys.argv[1])
        result = factorial(num)
        
        if result is None:
            print(json.dumps({"error": "Negative numbers are not supported"}))
            sys.exit(1)

        print(json.dumps({"factorial": result})) 
        sys.stdout.flush()

    except ValueError:
        print(json.dumps({"error": "Invalid input: must be an integer"}))
        sys.exit(1)