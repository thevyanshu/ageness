# ruff: noqa: E501 — scenario content strings exceed 100 chars intentionally

from __future__ import annotations

from typing import Any

from benchmark.models import ConversationTurn, InjectedFact, Scenario


class ConversationGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._turn_counter = 0
        self._rng = self._make_rng(seed)

    def _make_rng(self, seed: int) -> Any:
        import random
        return random.Random(seed)

    def reset(self) -> None:
        self._turn_counter = 0

    def _next_turn(self) -> int:
        self._turn_counter += 1
        return self._turn_counter

    def _fact(
        self, tid: int, fid: str, content: str,
        category: str = "decision", recall_at: list[int] | None = None,
    ) -> InjectedFact:
        return InjectedFact(
            fact_id=fid,
            turn_id=tid,
            content=content,
            category=category,
            expected_recall_turns=recall_at or [],
        )

    def generate_architecture_planning(self) -> Scenario:
        self.reset()
        turns: list[ConversationTurn] = []
        facts: list[InjectedFact] = []

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="We need to design a new microservice architecture for our e-commerce platform. "
                    "Currently we have a monolith. Requirements: handle 10k RPM, support multi-region "
                    "deployment, enable team autonomy.",
        ))
        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="assistant",
            content="I recommend starting with domain-driven design to identify bounded contexts. "
                    "Key domains: Product Catalog, Order Management, Payment, Shipping, User. "
                    "I suggest event-driven communication via Kafka.",
        ))
        facts.append(self._fact(tid, "domain_driven_design",
                                "Use domain-driven design with bounded contexts",
                                "decision", recall_at=[20, 50]))

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="What about the database? Should each service have its own database?",
        ))
        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="assistant",
            content="Yes, each microservice gets its own database. This ensures loose coupling. "
                    "Use PostgreSQL for transactional services and MongoDB for catalog. "
                    "Important: we must handle distributed transactions carefully — use Saga pattern.",
        ))
        facts.append(self._fact(tid, "database_per_service",
                                "Each microservice has its own database",
                                "decision", recall_at=[15, 30, 60]))
        facts.append(self._fact(tid, "saga_pattern",
                                "Use Saga pattern for distributed transactions",
                                "decision", recall_at=[25, 55]))

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="The payment team wants to use Stripe. Good choice?",
        ))
        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="assistant",
            content="Stripe is excellent for payment processing. Use Stripe Connect for multi-region "
                    "payouts. Store only the payment intent IDs to minimize PCI scope. "
                    "Implement idempotency keys for retry safety.",
        ))
        facts.append(self._fact(tid, "stripe_payment",
                                "Use Stripe Connect with idempotency keys",
                                "decision", recall_at=[10, 40]))

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="Wait — I'm worried about the complexity. How do we handle service discovery?",
        ))
        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="assistant",
            content="Use Kubernetes-native service discovery. Each service registers via label "
                    "selectors. For resilience, implement circuit breakers with exponential backoff. "
                    "Also use a service mesh (Istio) for observability and traffic management.",
        ))
        facts.append(self._fact(tid, "service_discovery",
                                "Kubernetes-native service discovery with Istio",
                                "decision", recall_at=[18, 45]))

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="We have an existing MySQL database with 500GB of data. How do we migrate?",
        ))
        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="assistant",
            content="Use the strangler fig pattern. Extract one domain at a time, starting with "
                    "Product Catalog (fewest cross-dependencies). Use Change Data Capture (Debezium) "
                    "to sync during transition. We must maintain 99.95% uptime during migration.",
        ))
        facts.append(self._fact(tid, "strangler_fig",
                                "Use strangler fig pattern with Debezium CDC",
                                "decision", recall_at=[12, 35]))
        facts.append(self._fact(tid, "uptime_constraint",
                                "Must maintain 99.95% uptime during migration",
                                "constraint", recall_at=[22, 48]))

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="Great plan. Let's start implementing. Initial sprint: set up event schemas.",
        ))

        return Scenario(
            scenario_id="arch_planning_001",
            title="E-commerce Microservice Architecture Planning",
            description="Full architecture planning session for migrating a monolith to microservices",
            turns=turns,
            facts=facts,
            expected_decisions=[
                "Use domain-driven design with bounded contexts",
                "Kafka for event-driven communication",
                "Database per service (PostgreSQL + MongoDB)",
                "Saga pattern for distributed transactions",
                "Stripe Connect with idempotency keys",
                "Kubernetes-native service discovery with Istio",
                "Strangler fig pattern with Debezium CDC",
            ],
            expected_goals=[
                "Handle 10k RPM",
                "Support multi-region deployment",
                "Enable team autonomy",
                "Maintain 99.95% uptime during migration",
            ],
            query_pairs=[
                {"turn": 20, "query": "What communication pattern did we choose?",
                 "expected": "Kafka"},
                {"turn": 30, "query": "Why did we pick database per service?",
                 "expected": "loose coupling"},
                {"turn": 50, "query": "How do we migrate from MySQL?",
                 "expected": "strangler fig"},
                {"turn": 60, "query": "What was the uptime constraint?",
                 "expected": "99.95"},
            ],
        )

    def generate_debugging_session(self, num_bugs: int = 5) -> Scenario:
        self.reset()
        turns: list[ConversationTurn] = []
        facts: list[InjectedFact] = []

        bug_patterns = [
            ("memory_leak",
             "Memory leak in connection pool — connections not released on timeout"),
            ("race_condition",
             "Race condition in order status update when payment and shipping webhooks fire"),
            ("cache_staleness",
             "Redis cache invalidation misses on product price updates under high write volume"),
            ("auth_timeout",
             "OAuth token refresh timing out when 50+ concurrent users authenticate"),
            ("db_deadlock",
             "PostgreSQL deadlock on concurrent order inserts in peak hours"),
        ]

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="We're seeing production issues. Let's debug systematically. "
                    "First issue: connection pool exhaustion after recent deploy.",
        ))
        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="assistant",
            content=f"Let's investigate the {bug_patterns[0][1]}. Check connection timeout "
                    f"settings and ensure connections return to pool in all error paths.",
        ))
        facts.append(self._fact(tid, bug_patterns[0][0], bug_patterns[0][1],
                                "bug", recall_at=[10]))

        for i, (bug_id, desc) in enumerate(bug_patterns[1:], 1):
            tid = self._next_turn()
            turns.append(ConversationTurn(
                turn_id=tid, role="user",
                content=f"Next issue: we found {bug_id.replace('_', ' ')}.",
            ))
            tid = self._next_turn()
            turns.append(ConversationTurn(
                turn_id=tid, role="assistant",
                content=f"Analyzing {desc}. Recommended fix: implement proper isolation. "
                        f"This is a critical path issue.",
            ))
            facts.append(self._fact(tid, bug_id, desc, "bug",
                                    recall_at=[5 * (i + 2)]))

        tid = self._next_turn()
        turns.append(ConversationTurn(
            turn_id=tid, role="user",
            content="Great debugging session. Let's create tickets and prioritize.",
        ))

        return Scenario(
            scenario_id="debug_session_001",
            title="Production Bug Diagnosis Marathon",
            description=f"Debugging {num_bugs} production issues across multiple services",
            turns=turns,
            facts=facts,
            expected_decisions=[
                "Fix connection pool timeout handling",
                "Add distributed lock for order status updates",
                "Implement write-through cache invalidation",
                "Add connection queuing for OAuth refresh",
                "Use advisory locks for concurrent order inserts",
            ],
            query_pairs=[
                {"turn": 15, "query": "What was the first bug?",
                 "expected": "connection pool"},
                {"turn": 25, "query": "How many bugs did we find?",
                 "expected": str(num_bugs)},
            ],
        )

    def generate_long_session(self, num_turns: int = 100) -> Scenario:
        self.reset()
        turns: list[ConversationTurn] = []
        facts: list[InjectedFact] = []

        topics = [
            ("auth",
             "Implement OAuth2 with PKCE flow using Auth0 as the identity provider"),
            ("cache",
             "Use Redis cluster with read-replicas for session caching"),
            ("queue",
             "Set up RabbitMQ with dead-letter queues for async job processing"),
            ("monitoring",
             "Deploy Prometheus + Grafana with custom business metrics"),
            ("logging",
             "Implement structured JSON logging with OpenTelemetry tracing"),
            ("deploy",
             "Set up GitHub Actions CI/CD with blue-green deployment strategy"),
            ("security",
             "Run monthly penetration testing and dependency vulnerability scans"),
            ("compliance",
             "Achieve SOC2 compliance with audit logging and access controls"),
        ]

        for i in range(num_turns // 2):
            topic = topics[i % len(topics)]
            tid = self._next_turn()
            content = topic[1]
            if i > 0:
                content = f"{topic[1]} — discuss trade-offs and implementation plan."
            turns.append(ConversationTurn(
                turn_id=tid, role="user",
                content=f"Let's work on {content}",
            ))
            tid = self._next_turn()
            turns.append(ConversationTurn(
                turn_id=tid, role="assistant",
                content=f"Good approach for {topic[0]}. Key considerations: "
                        f"scalability, fault tolerance, and operational overhead.",
            ))
            facts.append(self._fact(
                tid, f"topic_{topic[0]}_{i}", content,
                topic[0], recall_at=[i + 10, i + 50],
            ))

        expected_decisions = [f.content for f in facts]

        return Scenario(
            scenario_id=f"long_session_{num_turns}",
            title=f"Long-Running Session ({num_turns} turns)",
            description="Extended conversation with multiple topics",
            turns=turns,
            facts=facts,
            expected_decisions=expected_decisions,
        )
