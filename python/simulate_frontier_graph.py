#!/usr/bin/env python3
"""Evaluate degree-capped Delaunay frontier graphs on random door locations."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError


Point = tuple[int, int]
Edge = tuple[int, int]


@dataclass
class TrialMetrics:
    connected: bool
    component_count: int
    largest_component_fraction: float
    initial_edge_count: int
    final_edge_count: int
    initial_max_degree: int
    final_max_degree: int
    degree_utilization: float
    qhull_jittered: bool


def canonical_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def all_door_midpoints(width: int, height: int) -> list[Point]:
    # Doubled coordinates represent exact half-tile positions without floats.
    vertical = [(2 * x, 2 * y + 1) for x in range(width + 1) for y in range(height)]
    horizontal = [(2 * x + 1, 2 * y) for x in range(width) for y in range(height + 1)]
    return vertical + horizontal


def sample_uniform(rng: random.Random, candidates: list[Point], count: int) -> list[Point]:
    return rng.sample(candidates, count)


def sample_clustered(
    rng: random.Random,
    width: int,
    height: int,
    count: int,
    cluster_count: int,
    cluster_sigma: float,
) -> list[Point]:
    centers = [
        (rng.uniform(0, width), rng.uniform(0, height))
        for _ in range(cluster_count)
    ]
    points: set[Point] = set()
    max_attempts = count * 100
    for _ in range(max_attempts):
        if len(points) == count:
            break
        center_x, center_y = rng.choice(centers)
        tile_x = min(max(round(rng.gauss(center_x, cluster_sigma)), 0), width - 1)
        tile_y = min(max(round(rng.gauss(center_y, cluster_sigma)), 0), height - 1)
        side = rng.randrange(4)
        if side == 0:
            point = (2 * tile_x, 2 * tile_y + 1)
        elif side == 1:
            point = (2 * tile_x + 2, 2 * tile_y + 1)
        elif side == 2:
            point = (2 * tile_x + 1, 2 * tile_y)
        else:
            point = (2 * tile_x + 1, 2 * tile_y + 2)
        points.add(point)
    if len(points) != count:
        raise ValueError(
            f"could not sample {count} unique clustered door locations; "
            "increase the map size or cluster sigma"
        )
    return sorted(points)


def delaunay_edges(points: list[Point]) -> tuple[set[Edge], bool]:
    if len(points) < 2:
        return set(), False
    if len(points) == 2:
        return {(0, 1)}, False

    coordinates = np.asarray(points, dtype=np.float64)
    jittered = False
    try:
        triangulation = Delaunay(coordinates)
    except QhullError:
        # QJ handles lattice degeneracies such as collinear or cocircular points.
        triangulation = Delaunay(coordinates, qhull_options="Qbb Qc Qz Q12 QJ")
        jittered = True

    edges = set()
    for simplex in triangulation.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a = int(simplex[i])
                b = int(simplex[j])
                if a < len(points) and b < len(points):
                    edges.add(canonical_edge(a, b))
    return edges, jittered


def edge_length_squared(points: list[Point], edge: Edge) -> int:
    a, b = edge
    dx = points[a][0] - points[b][0]
    dy = points[a][1] - points[b][1]
    return dx * dx + dy * dy


def prune_edges(points: list[Point], initial_edges: set[Edge], max_degree: int) -> set[Edge]:
    edges = set(initial_edges)
    neighbors = [set() for _ in points]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    while True:
        excess_vertices = [vertex for vertex, adjacent in enumerate(neighbors) if len(adjacent) > max_degree]
        if not excess_vertices:
            return edges
        vertex = min(excess_vertices, key=lambda value: (-len(neighbors[value]), value))
        neighbor = min(
            neighbors[vertex],
            key=lambda value: (
                -len(neighbors[value]),
                -edge_length_squared(points, canonical_edge(vertex, value)),
                value,
            ),
        )
        edge = canonical_edge(vertex, neighbor)
        edges.remove(edge)
        neighbors[vertex].remove(neighbor)
        neighbors[neighbor].remove(vertex)


def component_sizes(vertex_count: int, edges: set[Edge]) -> list[int]:
    neighbors = [[] for _ in range(vertex_count)]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    sizes = []
    unseen = set(range(vertex_count))
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        size = 0
        while queue:
            vertex = queue.popleft()
            size += 1
            for neighbor in neighbors[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    return sizes


def degree_sequence(vertex_count: int, edges: set[Edge]) -> list[int]:
    degrees = [0] * vertex_count
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees


def run_trial(points: list[Point], max_degree: int) -> TrialMetrics:
    initial_edges, jittered = delaunay_edges(points)
    edges = prune_edges(points, initial_edges, max_degree)
    components = component_sizes(len(points), edges)
    initial_degrees = degree_sequence(len(points), initial_edges)
    degrees = degree_sequence(len(points), edges)
    return TrialMetrics(
        connected=len(components) == 1,
        component_count=len(components),
        largest_component_fraction=max(components) / len(points),
        initial_edge_count=len(initial_edges),
        final_edge_count=len(edges),
        initial_max_degree=max(initial_degrees, default=0),
        final_max_degree=max(degrees, default=0),
        degree_utilization=sum(degrees) / (len(points) * max_degree),
        qhull_jittered=jittered,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def print_summary(distribution: str, point_count: int, max_degree: int, metrics: list[TrialMetrics]) -> None:
    failures = sum(not metric.connected for metric in metrics)
    print(
        f"{distribution:9} n={point_count:3} K={max_degree:2} "
        f"connected={1 - failures / len(metrics):7.2%} "
        f"components={mean([metric.component_count for metric in metrics]):5.2f} "
        f"largest={mean([metric.largest_component_fraction for metric in metrics]):7.2%} "
        f"edges={mean([metric.final_edge_count for metric in metrics]):6.1f} "
        f"util={mean([metric.degree_utilization for metric in metrics]):7.2%} "
        f"initial-max={mean([metric.initial_max_degree for metric in metrics]):5.2f} "
        f"jitter={sum(metric.qhull_jittered for metric in metrics):4}/{len(metrics)}"
    )


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-width", type=int, default=72)
    parser.add_argument("--map-height", type=int, default=72)
    parser.add_argument("--point-counts", type=parse_int_list, default=parse_int_list("8,16,32,64,128"))
    parser.add_argument("--degrees", type=parse_int_list, default=parse_int_list("2,3,4,5,6,8"))
    parser.add_argument("--distributions", choices=["uniform", "clustered"], nargs="+", default=["uniform", "clustered"])
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument("--cluster-sigma", type=float, default=4.0)
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be greater than zero")
    if args.map_width <= 0 or args.map_height <= 0:
        parser.error("map dimensions must be greater than zero")
    if any(count <= 0 for count in args.point_counts):
        parser.error("--point-counts values must be greater than zero")
    if any(degree <= 0 for degree in args.degrees):
        parser.error("--degrees values must be greater than zero")

    candidates = all_door_midpoints(args.map_width, args.map_height)
    if max(args.point_counts) > len(candidates):
        parser.error("point count exceeds the number of unique door locations")

    rng = random.Random(args.seed)
    for distribution in args.distributions:
        for point_count in args.point_counts:
            samples = []
            for _ in range(args.trials):
                if distribution == "uniform":
                    samples.append(sample_uniform(rng, candidates, point_count))
                else:
                    samples.append(
                        sample_clustered(
                            rng,
                            args.map_width,
                            args.map_height,
                            point_count,
                            args.cluster_count,
                            args.cluster_sigma,
                        )
                    )
            for max_degree in args.degrees:
                print_summary(
                    distribution,
                    point_count,
                    max_degree,
                    [run_trial(points, max_degree) for points in samples],
                )


if __name__ == "__main__":
    main()
