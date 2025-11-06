from openai import OpenAI 
client = OpenAI() 

#run hpx prompt
#GPT5 doesn't support temperature
prompt = "#include <hpx/hpx_main.hpp>\n\n/* Factorize the matrix A into A=LU where L is a lower triangular matrix and U is an upper triangular matrix.\n   Store the results for L and U into the original matrix A. \n   A is an NxN matrix stored in row-major.\n   Use HPX to compute in parallel. Assume main() and hpx_main() functions have already been initialized and simply complete the function below.\n   Example:\n\n   input: [[4, 3], [6, 3]]\n   output: [[4, 3], [1.5, -1.5]]\n*/\nvoid luFactorize(std::vector<double> &A, size_t N) {"
response = client.responses.create( 
    model ="gpt-4"    
    , input = f"{prompt}"
    , max_output_tokens= 2048 
    , temperature=0.95
)
#get output text
print(response.output_text)