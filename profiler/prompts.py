"""48 hardcoded test prompts for the Clairvoyant Scheduler profiler.

Target distribution
-------------------
SHORT   17 prompts  (~35%)  ≤ ~15 tokens  quick factual / one-liner
MEDIUM  12 prompts  (~25%)  16–50 tokens  constrained or multi-sentence
LONG    19 prompts  (~40%)  51+ tokens    detailed specs / multi-step

Total: 48  →  approx 35 % short · 25 % medium · 40 % long
"""

# ---------------------------------------------------------------------------
# SHORT  (17 prompts)
# ---------------------------------------------------------------------------

SHORT: list[str] = [
    # original 7
    "What is a neural network?",
    "Define recursion.",
    "List five sorting algorithms.",
    "What does HTTP stand for?",
    "Explain what a pointer is in C.",
    "How does a hash table work?",
    "What is the difference between TCP and UDP?",
    # Fix-1 additions
    "What is 2+2?",
    "Name the capital of Japan.",
    "What color is the sky?",
    "How many days in a week?",
    "What does CPU stand for?",
    # 5 more factual
    "Who wrote Hamlet?",
    "What does RAM stand for?",
    "How many planets are in our solar system?",
    "What is the boiling point of water in Celsius?",
    "What does SQL stand for?",
]

# ---------------------------------------------------------------------------
# MEDIUM  (12 prompts)
# ---------------------------------------------------------------------------

MEDIUM: list[str] = [
    # original 7
    "Write a Python function that checks whether a given string is a palindrome. Include a docstring and one example.",
    "Explain the concept of gradient descent in machine learning in under 100 words.",
    "Compare bubble sort and merge sort. Which is better for large datasets and why?",
    "Summarize the CAP theorem in two sentences.",
    "Write a SQL query that returns the top 5 customers by total order value from an 'orders' table.",
    "What are the pros and cons of using microservices over a monolithic architecture?",
    "Implement a basic stack in Python using a list. Include push, pop, and peek methods.",
    # 5 new — summary/constraint focused
    "Summarize the purpose of Docker in one sentence.",
    "Explain the difference between a stack and a queue in under 50 words.",
    "What is a REST API? Explain in exactly three bullet points.",
    "Describe the map-reduce pattern briefly with one real-world example.",
    "List three advantages of TypeScript over JavaScript. Keep each point to one sentence.",
]

# ---------------------------------------------------------------------------
# LONG  (19 prompts)
# ---------------------------------------------------------------------------

LONG: list[str] = [
    # original 6
    (
        "I am building a REST API in FastAPI for a task management application. "
        "The app has users, projects, and tasks. Each task belongs to a project, "
        "and each project belongs to a user. Write the SQLAlchemy models for these "
        "three entities, including relationships, and a Pydantic schema for creating "
        "a new task. Use best practices."
    ),
    (
        "Explain the transformer architecture from the 'Attention Is All You Need' paper. "
        "Cover: the encoder/decoder structure, multi-head self-attention, positional "
        "encoding, and why the architecture replaced RNNs for sequence-to-sequence tasks. "
        "Assume the reader has a solid ML background."
    ),
    (
        "I have a Python script that reads a large CSV file (10 GB) row by row, "
        "performs some regex transformations, and writes the result to a new CSV. "
        "It currently takes 4 hours to run. Suggest at least three concrete optimizations "
        "I can apply, with example code snippets for each."
    ),
    (
        "Write a detailed technical design document outline for a distributed job scheduler. "
        "Include sections for: system goals, high-level architecture, data model, "
        "API contract, fault tolerance strategy, and open questions."
    ),
    (
        "Given the following Python class that implements a Least Recently Used (LRU) cache "
        "using a dictionary and a doubly linked list, identify any bugs, suggest refactors "
        "for readability, and add type annotations:\n\n"
        "class LRUCache:\n"
        "    def __init__(self, cap):\n"
        "        self.cap = cap\n"
        "        self.cache = {}\n"
        "        self.head, self.tail = Node(0,0), Node(0,0)\n"
        "        self.head.next = self.tail\n"
        "        self.tail.prev = self.head\n"
        "    def get(self, key):\n"
        "        if key in self.cache:\n"
        "            self.remove(self.cache[key])\n"
        "            self.insert(self.cache[key])\n"
        "            return self.cache[key].val\n"
        "        return -1\n"
    ),
    (
        "I am a researcher studying how large language models handle scheduling problems. "
        "Describe three experimental setups I could use to measure whether an LLM can "
        "accurately predict the execution time of a task given only its natural language "
        "description. For each setup, describe the dataset, the model inputs and outputs, "
        "the evaluation metric, and at least one potential confounder."
    ),
    # 13 new long prompts
    (
        "Design a rate-limiting system for a public REST API that serves 50 million requests "
        "per day. Cover: the algorithm choice (token bucket vs leaky bucket vs sliding window), "
        "where to enforce limits (API gateway, middleware, or service layer), how to store "
        "counters at scale, how to handle distributed deployments across multiple regions, "
        "and what HTTP response headers and status codes to return to clients."
    ),
    (
        "I am training a binary text classifier using a fine-tuned BERT model on an imbalanced "
        "dataset (95% negative, 5% positive). The model achieves 95% accuracy but terrible "
        "recall on the positive class. Walk me through: (1) why accuracy is a misleading metric "
        "here, (2) three resampling or loss-weighting techniques I can apply, (3) which "
        "evaluation metrics I should switch to, and (4) how to tune the decision threshold "
        "after training."
    ),
    (
        "I have a PostgreSQL table 'events' with 200 million rows and the following columns: "
        "id SERIAL, user_id INT, event_type VARCHAR(64), created_at TIMESTAMPTZ, payload JSONB. "
        "A query filtering by user_id and event_type over the last 30 days takes 8 seconds. "
        "Diagnose the likely bottlenecks and propose at least four concrete optimizations, "
        "including index strategy, partitioning approach, and any schema changes."
    ),
    (
        "Conduct a security review of the following Python Flask endpoint. Identify every "
        "vulnerability class present, explain the attack vector for each, and provide a "
        "corrected version of the code:\n\n"
        "@app.route('/login', methods=['POST'])\n"
        "def login():\n"
        "    username = request.form['username']\n"
        "    password = request.form['password']\n"
        "    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"\n"
        "    user = db.execute(query).fetchone()\n"
        "    if user:\n"
        "        session['user'] = username\n"
        "        return redirect('/')\n"
        "    return 'Invalid credentials', 401\n"
    ),
    (
        "I need to design the data model for a multi-tenant SaaS application where each "
        "tenant has its own users, subscription plan, and isolated data. Compare three "
        "multi-tenancy database strategies: (1) shared schema with tenant_id column, "
        "(2) separate schema per tenant, (3) separate database per tenant. For each, "
        "discuss storage cost, query isolation, migration complexity, and when you would "
        "choose it."
    ),
    (
        "Write a Python asyncio producer-consumer pipeline where: the producer reads lines "
        "from a large file asynchronously, puts them into an asyncio.Queue with a max size "
        "of 100, three consumer coroutines pull from the queue, apply a regex transform, "
        "and write results to an output file. Include proper shutdown logic so all consumers "
        "drain the queue before exiting. Add type hints throughout."
    ),
    (
        "I am migrating a monolithic Django application to a microservices architecture. "
        "The monolith has these modules: user management, order processing, inventory, "
        "and notifications. Propose a migration strategy using the strangler-fig pattern. "
        "For each service, define its bounded context, the API contract it exposes, "
        "how it communicates with others (sync vs async), and the order in which services "
        "should be extracted to minimise risk."
    ),
    (
        "Build a Python context manager called Timer that: measures wall-clock time of any "
        "code block, supports an optional label string for logging, prints elapsed time on "
        "exit in milliseconds, and raises a TimeoutError if the block exceeds an optional "
        "max_seconds threshold. Show the implementation and three usage examples covering "
        "the normal case, the timeout case, and nesting two timers."
    ),
    (
        "Explain how garbage collection works in CPython. Cover: reference counting and "
        "why it is not sufficient alone, the cyclic garbage collector and the three "
        "generational heaps, what triggers a collection cycle, the cost of gc.collect() "
        "in a latency-sensitive application, and two practical strategies to reduce GC "
        "pressure in long-running Python services."
    ),
    (
        "I want to set up a CI/CD pipeline for a Python microservice using GitHub Actions. "
        "The pipeline should: run unit tests with pytest and fail on < 80% coverage, "
        "lint with ruff and type-check with mypy, build and push a Docker image to GHCR "
        "with a SHA tag on every merge to main, and deploy to a Kubernetes cluster using "
        "kubectl rollout. Write the complete GitHub Actions YAML for this workflow."
    ),
    (
        "Describe the full lifecycle of an HTTP/2 request from the moment a browser "
        "sends it to when it renders the response. Include: TLS handshake and ALPN "
        "negotiation, stream multiplexing and flow control, header compression with HPACK, "
        "how server push works and when it helps, and how HTTP/2 differs from HTTP/1.1 "
        "for a page with 40 assets."
    ),
    (
        "I have a React application where a parent component fetches a list of 1000 items "
        "and passes it as a prop to a child component that renders them in a table. "
        "The page becomes sluggish when filtering. Identify all the likely performance "
        "problems in this setup and provide concrete fixes using useMemo, useCallback, "
        "React.memo, and windowing (react-window). Include annotated code examples for each."
    ),
    (
        "Write a Kubernetes Deployment and Service manifest for a stateless Python FastAPI "
        "application. Requirements: 3 replicas, rolling update strategy with maxSurge=1 and "
        "maxUnavailable=0, resource requests of 256Mi / 0.25 CPU and limits of 512Mi / 0.5 CPU, "
        "a liveness probe on GET /healthz and a readiness probe on GET /ready, "
        "an environment variable API_KEY injected from a Kubernetes Secret, "
        "and a ClusterIP Service on port 80 targeting container port 8000."
    ),
]

# ---------------------------------------------------------------------------
# Aggregate + validate
# ---------------------------------------------------------------------------

ALL: list[str] = SHORT + MEDIUM + LONG

_expected = {"SHORT": 17, "MEDIUM": 12, "LONG": 19, "ALL": 48}
_actual   = {"SHORT": len(SHORT), "MEDIUM": len(MEDIUM), "LONG": len(LONG), "ALL": len(ALL)}
assert _actual == _expected, f"Prompt count mismatch: expected {_expected}, got {_actual}"

DISTRIBUTION = {
    "short_pct":  round(len(SHORT) / len(ALL) * 100, 1),
    "medium_pct": round(len(MEDIUM) / len(ALL) * 100, 1),
    "long_pct":   round(len(LONG)  / len(ALL) * 100, 1),
}
