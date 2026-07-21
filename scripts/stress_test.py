import asyncio
import httpx
import time
import sys

# Reconfigure stdout to utf-8 if possible, or just strip emojis
API_URL = "http://127.0.0.1:8000/ask"
CONCURRENT_REQUESTS = 20

async def fetch(client, i):
    payload = {"query": f"What is OneInbox? (Request {i})"}
    start_time = time.time()
    try:
        response = await client.post(API_URL, json=payload, timeout=30.0)
        elapsed = time.time() - start_time
        return i, response.status_code, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return i, str(e), elapsed

async def main():
    print(f"Starting Stress Test: Firing {CONCURRENT_REQUESTS} concurrent requests to the pipeline...")
    start_total = time.time()
    
    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)
        
    total_time = time.time() - start_total
    
    success_count = 0
    rate_limited_count = 0
    error_count = 0
    
    print("\n--- STRESS TEST RESULTS ---")
    for i, status, elapsed in results:
        if status == 200:
            success_count += 1
        elif status == 429:
            rate_limited_count += 1
        else:
            error_count += 1
            
    print(f"Total Time Taken: {total_time:.2f} seconds")
    print(f"Successful Requests (HTTP 200): {success_count}")
    print(f"Rate Limited (HTTP 429): {rate_limited_count} (Proves OWASP Compliance stub!)")
    print(f"Failed/Timeout Errors: {error_count}")
    
    if rate_limited_count > 0:
        print("\nNOTE: The system successfully blocked excessive traffic using the Rate Limiter, preventing LLM cost blowouts!")
    elif success_count == CONCURRENT_REQUESTS:
        print("\nNOTE: All requests succeeded. The pipeline can handle this concurrency gracefully!")

if __name__ == "__main__":
    asyncio.run(main())
