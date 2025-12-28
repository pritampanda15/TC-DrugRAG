"""
Temporal Causal Knowledge Graph Implementation.

This is the core data structure that stores entities and their temporal
causal relationships for drug discovery.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Generator, Optional

import networkx as nx
from neo4j import GraphDatabase

from .schemas import (
    CausalPath,
    CausalRelationType,
    Node,
    NodeType,
    TemporalEdge,
)

logger = logging.getLogger(__name__)


class TemporalCausalKG:
    """
    Temporal Causal Knowledge Graph for Drug Discovery.

    This class manages a knowledge graph where edges have temporal validity
    windows and causal relationship types. It supports both in-memory
    (NetworkX) and database-backed (Neo4j) storage.

    Key Features:
    - Temporal edge validity tracking
    - Causal path enumeration with temporal filtering
    - Confidence decay over time
    - Multi-hop causal reasoning

    Example:
        >>> kg = TemporalCausalKG(backend="networkx")
        >>> drug = Node(id="DB00001", name="Metformin", node_type=NodeType.DRUG)
        >>> target = Node(id="PRKAA1", name="AMPK", node_type=NodeType.PROTEIN)
        >>> kg.add_node(drug)
        >>> kg.add_node(target)
        >>> edge = TemporalEdge(
        ...     source=drug, target=target,
        ...     relation_type=CausalRelationType.ACTIVATES,
        ...     confidence=0.95,
        ...     t_start=datetime(2000, 1, 1)
        ... )
        >>> kg.add_edge(edge)
    """

    def __init__(
        self,
        backend: str = "networkx",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "tcdrugrag",
    ):
        """
        Initialize the Temporal Causal Knowledge Graph.

        Args:
            backend: Storage backend ("networkx" or "neo4j")
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.backend = backend

        if backend == "networkx":
            self._graph = nx.MultiDiGraph()
            self._nodes: dict[str, Node] = {}
            self._edges: list[TemporalEdge] = []
            self._edge_index: dict[str, list[TemporalEdge]] = defaultdict(list)

        elif backend == "neo4j":
            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j credentials required for neo4j backend")
            self._driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            self._database = neo4j_database
            self._init_neo4j_schema()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"Initialized TemporalCausalKG with {backend} backend")

    def _init_neo4j_schema(self):
        """Initialize Neo4j schema with indexes and constraints."""
        with self._driver.session(database=self._database) as session:
            # Create indexes for node types
            for node_type in NodeType:
                session.run(f"""
                    CREATE INDEX IF NOT EXISTS FOR (n:{node_type.value})
                    ON (n.id)
                """)

            # Create index for temporal edges
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR ()-[r:CAUSAL_RELATION]-()
                ON (r.t_start, r.t_end)
            """)

            logger.info("Neo4j schema initialized")

    def add_node(self, node: Node) -> None:
        """
        Add a node to the knowledge graph.

        Args:
            node: The node to add
        """
        if self.backend == "networkx":
            self._nodes[node.id] = node
            self._graph.add_node(
                node.id,
                name=node.name,
                node_type=node.node_type.value,
                properties=node.properties,
                external_ids=node.external_ids,
            )
        else:
            with self._driver.session(database=self._database) as session:
                session.run(f"""
                    MERGE (n:{node.node_type.value} {{id: $id}})
                    SET n.name = $name,
                        n.properties = $properties,
                        n.external_ids = $external_ids
                """,
                    id=node.id,
                    name=node.name,
                    properties=node.properties,
                    external_ids=node.external_ids,
                )

    def add_edge(self, edge: TemporalEdge) -> None:
        """
        Add a temporal causal edge to the knowledge graph.

        Args:
            edge: The temporal edge to add
        """
        # Ensure nodes exist
        self.add_node(edge.source)
        self.add_node(edge.target)

        if self.backend == "networkx":
            self._edges.append(edge)
            self._edge_index[edge.source.id].append(edge)

            self._graph.add_edge(
                edge.source.id,
                edge.target.id,
                relation_type=edge.relation_type.value,
                confidence=edge.confidence,
                t_start=edge.t_start.isoformat(),
                t_end=edge.t_end.isoformat() if edge.t_end else None,
                evidence_type=edge.evidence_type.value,
                evidence_sources=edge.evidence_sources,
            )
        else:
            with self._driver.session(database=self._database) as session:
                session.run("""
                    MATCH (s {id: $source_id}), (t {id: $target_id})
                    CREATE (s)-[r:CAUSAL_RELATION {
                        relation_type: $relation_type,
                        confidence: $confidence,
                        t_start: datetime($t_start),
                        t_end: $t_end,
                        evidence_type: $evidence_type,
                        evidence_sources: $evidence_sources
                    }]->(t)
                """,
                    source_id=edge.source.id,
                    target_id=edge.target.id,
                    relation_type=edge.relation_type.value,
                    confidence=edge.confidence,
                    t_start=edge.t_start.isoformat(),
                    t_end=edge.t_end.isoformat() if edge.t_end else None,
                    evidence_type=edge.evidence_type.value,
                    evidence_sources=edge.evidence_sources,
                )

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        if self.backend == "networkx":
            return self._nodes.get(node_id)
        else:
            with self._driver.session(database=self._database) as session:
                result = session.run("""
                    MATCH (n {id: $id})
                    RETURN n, labels(n) as labels
                """, id=node_id)
                record = result.single()
                if record:
                    node_data = dict(record["n"])
                    node_type = NodeType(record["labels"][0])
                    return Node(
                        id=node_data["id"],
                        name=node_data["name"],
                        node_type=node_type,
                        properties=node_data.get("properties", {}),
                        external_ids=node_data.get("external_ids", {}),
                    )
        return None

    def get_outgoing_edges(
        self,
        node_id: str,
        time_point: datetime = None,
        relation_types: list[CausalRelationType] = None,
        min_confidence: float = 0.0,
    ) -> list[TemporalEdge]:
        """
        Get all outgoing edges from a node, optionally filtered.

        Args:
            node_id: Source node ID
            time_point: Filter edges valid at this time
            relation_types: Filter by relation types
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching temporal edges
        """
        time_point = time_point or datetime.now()

        if self.backend == "networkx":
            edges = self._edge_index.get(node_id, [])

            filtered = []
            for edge in edges:
                if not edge.is_valid_at(time_point):
                    continue
                if relation_types and edge.relation_type not in relation_types:
                    continue
                if edge.get_temporal_confidence(time_point) < min_confidence:
                    continue
                filtered.append(edge)

            return filtered
        else:
            # Neo4j implementation
            query = """
                MATCH (s {id: $node_id})-[r:CAUSAL_RELATION]->(t)
                WHERE datetime($time_point) >= r.t_start
                  AND (r.t_end IS NULL OR datetime($time_point) <= r.t_end)
                  AND r.confidence >= $min_confidence
            """
            if relation_types:
                types_str = [rt.value for rt in relation_types]
                query += " AND r.relation_type IN $relation_types"

            query += " RETURN s, r, t"

            with self._driver.session(database=self._database) as session:
                params = {
                    "node_id": node_id,
                    "time_point": time_point.isoformat(),
                    "min_confidence": min_confidence,
                }
                if relation_types:
                    params["relation_types"] = types_str

                result = session.run(query, **params)
                # Convert to TemporalEdge objects
                # ... (implementation details)
                return []

    def find_causal_paths(
        self,
        source_id: str,
        target_id: str = None,
        max_depth: int = 4,
        time_point: datetime = None,
        min_path_confidence: float = 0.1,
        require_temporal_consistency: bool = True,
    ) -> Generator[CausalPath, None, None]:
        """
        Find all causal paths from source to target (or all reachable nodes).

        This is the core algorithm for causal retrieval - it enumerates
        paths through the knowledge graph following causal relationships.

        Args:
            source_id: Starting node ID
            target_id: Target node ID (optional, if None returns all paths)
            max_depth: Maximum path length
            time_point: Filter paths valid at this time
            min_path_confidence: Minimum cumulative path confidence
            require_temporal_consistency: Require causes to precede effects

        Yields:
            CausalPath objects representing valid paths
        """
        time_point = time_point or datetime.now()

        def _dfs(
            current_id: str,
            visited: set,
            current_path: list[TemporalEdge],
            depth: int,
        ) -> Generator[CausalPath, None, None]:
            """Depth-first search for causal paths."""
            if depth > max_depth:
                return

            # Check if we've reached the target
            if target_id and current_id == target_id:
                path = CausalPath(edges=list(current_path))
                if path.get_path_confidence(time_point) >= min_path_confidence:
                    if not require_temporal_consistency or path.is_temporally_consistent():
                        yield path
                return

            # If no target specified, yield all non-trivial paths
            if not target_id and len(current_path) > 0:
                path = CausalPath(edges=list(current_path))
                if path.get_path_confidence(time_point) >= min_path_confidence:
                    if not require_temporal_consistency or path.is_temporally_consistent():
                        yield path

            # Explore neighbors
            for edge in self.get_outgoing_edges(current_id, time_point):
                next_id = edge.target.id
                if next_id not in visited:
                    visited.add(next_id)
                    current_path.append(edge)

                    yield from _dfs(next_id, visited, current_path, depth + 1)

                    current_path.pop()
                    visited.remove(next_id)

        visited = {source_id}
        yield from _dfs(source_id, visited, [], 0)

    def get_subgraph(
        self,
        node_ids: list[str],
        include_paths: bool = True,
        max_path_length: int = 2,
    ) -> "TemporalCausalKG":
        """
        Extract a subgraph containing specified nodes and paths between them.

        Args:
            node_ids: List of node IDs to include
            include_paths: Whether to include paths between nodes
            max_path_length: Maximum path length for connecting nodes

        Returns:
            New TemporalCausalKG containing the subgraph
        """
        subgraph = TemporalCausalKG(backend="networkx")

        # Add all specified nodes
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                subgraph.add_node(node)

        # Add edges between nodes
        if include_paths:
            for i, source_id in enumerate(node_ids):
                for target_id in node_ids[i+1:]:
                    for path in self.find_causal_paths(
                        source_id, target_id, max_depth=max_path_length
                    ):
                        for edge in path.edges:
                            subgraph.add_edge(edge)

        return subgraph

    def update_edge_confidence(
        self,
        source_id: str,
        target_id: str,
        relation_type: CausalRelationType,
        new_confidence: float,
        evidence_source: str = None,
    ) -> None:
        """
        Update the confidence of an existing edge (for feedback loop).

        Uses Bayesian update when evidence_source is provided.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Relation type to update
            new_confidence: New confidence value or likelihood
            evidence_source: New evidence source identifier
        """
        if self.backend == "networkx":
            for edge in self._edge_index.get(source_id, []):
                if (edge.target.id == target_id and
                    edge.relation_type == relation_type):
                    # Bayesian update: P(H|E) ∝ P(E|H) * P(H)
                    prior = edge.confidence
                    likelihood = new_confidence
                    posterior = (likelihood * prior) / (
                        likelihood * prior + (1 - likelihood) * (1 - prior)
                    )
                    edge.confidence = posterior

                    if evidence_source:
                        edge.evidence_sources.append(evidence_source)

                    logger.info(
                        f"Updated edge {source_id}->{target_id}: "
                        f"{prior:.3f} -> {posterior:.3f}"
                    )
                    break

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        if self.backend == "networkx":
            return len(self._nodes)
        else:
            with self._driver.session(database=self._database) as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                return result.single()["count"]

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the graph."""
        if self.backend == "networkx":
            return len(self._edges)
        else:
            with self._driver.session(database=self._database) as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                return result.single()["count"]

    def get_statistics(self) -> dict:
        """Get summary statistics about the knowledge graph."""
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_types": {},
            "relation_types": {},
        }

        if self.backend == "networkx":
            for node in self._nodes.values():
                node_type = node.node_type.value
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

            for edge in self._edges:
                rel_type = edge.relation_type.value
                stats["relation_types"][rel_type] = stats["relation_types"].get(rel_type, 0) + 1

        return stats

    def close(self):
        """Close any open connections."""
        if self.backend == "neo4j":
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
