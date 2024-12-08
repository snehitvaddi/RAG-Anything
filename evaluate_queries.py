import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from rouge_score import rouge_scorer
import json
import pandas as pd
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Test cases with expected SQL queries
test_cases = {
    "What recalls happened in the year 1994?": {
        "expected_sql": "SELECT * FROM incidents_train WHERE year = 1994"
    },
    "List all recalls involving Listeria monocytogenes.": {
        "expected_sql": "SELECT * FROM incidents_train WHERE hazard = 'Listeria monocytogenes'"
    },
    "Which recalls were issued for 'meat, egg, and dairy products'?": {
        "expected_sql": "SELECT * FROM incidents_train WHERE product_category = 'meat, egg and dairy products'"
    },
    "What products were recalled in the US in July 1994?": {
        "expected_sql": "SELECT product FROM incidents_train WHERE country = 'US' AND year = 1994 AND month = 7"
    },
    "Which product was affected by plastic fragments?": {
        "expected_sql": "SELECT product FROM incidents_train WHERE hazard = 'plastic fragment'"
    },
    "What are the most common hazards reported in this dataset?": {
        "expected_sql": "SELECT hazard, COUNT(*) as count FROM incidents_train GROUP BY hazard ORDER BY count DESC"
    },
    "Provide examples of recalls due to foreign bodies.": {
        "expected_sql": "SELECT * FROM incidents_train WHERE hazard_category = 'foreign bodies'"
    },
    "What are the specific products affected by Listeria spp.?": {
        "expected_sql": "SELECT product FROM incidents_train WHERE hazard = 'listeria monocytogenes'"
    },
    "List all recalls involving chemical hazards, if any.": {
        "expected_sql": "SELECT * FROM incidents_train WHERE hazard_category = 'chemical'"
    },
    "What foreign body hazards are most frequently mentioned?": {
        "expected_sql": "SELECT hazard, COUNT(*) as count FROM incidents_train WHERE hazard_category = 'foreign bodies' GROUP BY hazard ORDER BY count DESC"
    },
    "Find the number of recalls per year and sort by the most recent year first.": {
        "expected_sql": "SELECT year, COUNT(*) as recall_count FROM incidents_train WHERE year REGEXP '^[0-9]+$' GROUP BY year ORDER BY recall_count DESC"
    },
    "List all unique countries with the number of recalls they issued.": {
        "expected_sql": "SELECT country, COUNT(*) as recall_count FROM incidents_train GROUP BY country ORDER BY recall_count DESC"
    },
    "Identify the top 5 products recalled the most frequently.": {
        "expected_sql": "SELECT product, COUNT(*) as recall_count FROM incidents_train GROUP BY product ORDER BY recall_count DESC LIMIT 5"
    },
    "Find recalls that mentioned both 'Listeria' and 'salmonella' in the hazard text.": {
        "expected_sql": "SELECT * FROM incidents_train WHERE hazard LIKE '%Listeria%' AND hazard LIKE '%salmonella%'"
    },
    "Determine the average number of recalls per month for each year.": {
        "expected_sql": "SELECT year, AVG(recall_count) as avg_recalls_per_month FROM (SELECT year, month, COUNT(*) as recall_count FROM incidents_train GROUP BY year, month) as monthly_recalls WHERE year REGEXP '^[0-9]+$' GROUP BY year ORDER BY year"
    },
    "Find the product categories most affected by recalls due to foreign bodies.": {
        "expected_sql": "SELECT product_category, COUNT(*) as recall_count FROM incidents_train WHERE hazard_category = 'foreign bodies' GROUP BY product_category ORDER BY recall_count DESC"
    },
    "Identify months with the highest recall counts across all years.": {
        "expected_sql": "SELECT month, COUNT(*) as recall_count FROM incidents_train GROUP BY month ORDER BY recall_count DESC"
    }
}


def initialize_chain():
    """Initialize the LLM chain"""
    try:
        db = SQLDatabase.from_uri(
            f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}",
            sample_rows_in_table_info=3
        )
        llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4",
            temperature=0.2
        )
        
        # Create chain with specific configuration
        chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            verbose=True,
            return_direct=False,
            return_intermediate_steps=True,
            use_query_checker=True
        )
        
        return chain
    except Exception as e:
        print(f"Error initializing chain: {str(e)}")
        return None

def clean_sql_query(sql):
    """Clean SQL query by removing markdown formatting and other artifacts"""
    if not sql:
        return ""
    
    # Remove markdown SQL formatting
    sql = re.sub(r'```sql\s*', '', sql)
    sql = re.sub(r'\s*```', '', sql)
    
    # Remove backticks from column names
    sql = sql.replace('`', '')
    
    # Clean up whitespace
    sql = ' '.join(sql.split())
    
    # Remove any trailing semicolons
    sql = sql.rstrip(';')
    
    return sql

def calculate_metrics(generated_sql, expected_sql):
    """Calculate ROUGE scores between generated and expected SQL"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Clean and normalize SQL queries
    generated_sql = clean_sql_query(generated_sql).lower()
    expected_sql = clean_sql_query(expected_sql).lower()
    
    print(f"\nComparing SQL queries:")
    print(f"Generated: {generated_sql}")
    print(f"Expected:  {expected_sql}")
    
    scores = scorer.score(generated_sql, expected_sql)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def run_evaluation():
    """Run the evaluation pipeline"""
    chain = initialize_chain()
    if not chain:
        print("Failed to initialize chain. Exiting...")
        return
    
    results = []
    total_questions = len(test_cases)
    
    print(f"\nStarting evaluation of {total_questions} questions...")
    
    for idx, (question, expected) in enumerate(test_cases.items(), 1):
        print(f"\nProcessing question {idx}/{total_questions}:")
        print(f"Q: {question}")
        
        try:
            # Get model response
            response = chain.invoke({"query": question})
            
            # Debug print the response
            print("\nRaw response:")
            print(response)
            
            # Extract SQL query from response
            generated_sql = None
            if isinstance(response, dict):
                # Try different ways to extract SQL
                if 'intermediate_steps' in response:
                    steps = response['intermediate_steps']
                    print("\nIntermediate steps:", steps)
                    
                    # Look for SQL in steps
                    for step in steps:
                        if isinstance(step, str):
                            if 'SELECT' in step.upper():
                                generated_sql = step
                                break
                
                # Alternative: check direct SQL field
                elif 'sql_query' in response:
                    generated_sql = response['sql_query']
                
                # Check result field for SQL
                elif 'result' in response:
                    result = str(response['result'])
                    if 'SELECT' in result.upper():
                        generated_sql = result
            
            if generated_sql:
                print(f"\nFound SQL query: {generated_sql}")
                # Clean the SQL query and calculate metrics
                cleaned_sql = clean_sql_query(generated_sql)
                metrics = calculate_metrics(cleaned_sql, expected['expected_sql'])
                
                results.append({
                    'question': question,
                    'generated_sql': cleaned_sql,
                    'expected_sql': expected['expected_sql'],
                    'rouge1': metrics['rouge1'],
                    'rouge2': metrics['rouge2'],
                    'rougeL': metrics['rougeL'],
                    'success': True,
                    'raw_response': str(response)  # Store raw response for debugging
                })
                print("✓ Success")
            else:
                print("✗ No SQL query found in response")
                results.append({
                    'question': question,
                    'success': False,
                    'error': 'No SQL query generated',
                    'raw_response': str(response)  # Store raw response for debugging
                })
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                'question': question,
                'success': False,
                'error': str(e)
            })
    
    # Calculate average metrics
    successful_results = [r for r in results if r.get('success', False)]
    print(f"\nSuccessful results: {len(successful_results)}/{total_questions}")
    
    if successful_results:
        avg_metrics = {
            'avg_rouge1': sum(r['rouge1'] for r in successful_results) / len(successful_results),
            'avg_rouge2': sum(r['rouge2'] for r in successful_results) / len(successful_results),
            'avg_rougeL': sum(r['rougeL'] for r in successful_results) / len(successful_results),
            'success_rate': len(successful_results) / len(results)
        }
    else:
        print("No successful results to calculate metrics!")
        avg_metrics = {
            'avg_rouge1': 0,
            'avg_rouge2': 0,
            'avg_rougeL': 0,
            'success_rate': 0
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'metrics': avg_metrics,
        'detailed_results': results,
        'summary': {
            'total_questions': total_questions,
            'successful_queries': len(successful_results),
            'failed_queries': total_questions - len(successful_results)
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nEvaluation completed and results saved!")
    print(f"Success rate: {avg_metrics['success_rate']*100:.1f}%")
    print(f"Average ROUGE-1: {avg_metrics['avg_rouge1']:.3f}")
    print(f"Average ROUGE-2: {avg_metrics['avg_rouge2']:.3f}")
    print(f"Average ROUGE-L: {avg_metrics['avg_rougeL']:.3f}")

if __name__ == "__main__":
    run_evaluation() 