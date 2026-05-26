const NO_COMPONENT: usize = usize::MAX;

#[derive(Debug, Default)]
pub struct SccDag {
    pub component_count: usize,
    reachability: Vec<bool>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ComponentMerge {
    pub merged_components: Vec<usize>,
    pub component_remap: Vec<usize>,
}

impl SccDag {
    pub fn clear(&mut self) {
        self.component_count = 0;
        self.reachability.clear();
    }

    pub fn add_component(&mut self) -> usize {
        let old_count = self.component_count;
        let new_count = old_count + 1;
        let mut reachability = vec![false; new_count * new_count];
        for row in 0..old_count {
            for col in 0..old_count {
                reachability[row * new_count + col] = self.reachability[row * old_count + col];
            }
        }
        reachability[old_count * new_count + old_count] = true;
        self.component_count = new_count;
        self.reachability = reachability;
        old_count
    }

    pub fn add_edge(
        &mut self,
        from_component: usize,
        to_component: usize,
    ) -> Option<ComponentMerge> {
        if from_component == to_component {
            return None;
        }
        if !self.can_reach(to_component, from_component) {
            self.connect_components(from_component, to_component);
            return None;
        }

        let mut merge_components = vec![];
        for component in 0..self.component_count {
            if self.can_reach(to_component, component) && self.can_reach(component, from_component)
            {
                merge_components.push(component);
            }
        }
        Some(self.merge_components(&merge_components))
    }

    fn merge_components(&mut self, merge_components: &[usize]) -> ComponentMerge {
        debug_assert!(!merge_components.is_empty());
        let target = merge_components.iter().copied().min().unwrap();
        let mut in_merge = vec![false; self.component_count];
        for &component in merge_components {
            in_merge[component] = true;
        }

        let mut predecessors = vec![];
        let mut successors = vec![];
        for (component, is_merged) in in_merge.iter().copied().enumerate() {
            if !is_merged
                && merge_components
                    .iter()
                    .any(|&merged| self.can_reach(component, merged))
            {
                predecessors.push(component);
            }
            if !is_merged
                && merge_components
                    .iter()
                    .any(|&merged| self.can_reach(merged, component))
            {
                successors.push(component);
            }
        }

        let old_count = self.component_count;
        let mut component_remap = vec![NO_COMPONENT; old_count];
        let mut new_count = 0;
        for (component, is_merged) in in_merge.iter().copied().enumerate() {
            if component == target || !is_merged {
                component_remap[component] = new_count;
                new_count += 1;
            } else {
                component_remap[component] = target;
            }
        }

        let old_reachability =
            std::mem::replace(&mut self.reachability, vec![false; new_count * new_count]);
        self.component_count = new_count;

        for from in 0..old_count {
            if in_merge[from] {
                continue;
            }
            for to in 0..old_count {
                if in_merge[to] {
                    continue;
                }
                if old_reachability[from * old_count + to] {
                    self.set_reachable(component_remap[from], component_remap[to]);
                }
            }
        }
        self.set_reachable(target, target);
        for predecessor in predecessors {
            let predecessor = component_remap[predecessor];
            self.set_reachable(predecessor, target);
            for &successor in &successors {
                self.set_reachable(predecessor, component_remap[successor]);
            }
        }
        for successor in successors {
            let successor = component_remap[successor];
            self.set_reachable(target, successor);
        }

        ComponentMerge {
            merged_components: merge_components.to_vec(),
            component_remap,
        }
    }

    fn connect_components(&mut self, from_component: usize, to_component: usize) {
        let mut predecessors = vec![];
        let mut successors = vec![];
        for component in 0..self.component_count {
            if self.can_reach(component, from_component) {
                predecessors.push(component);
            }
            if self.can_reach(to_component, component) {
                successors.push(component);
            }
        }
        for predecessor in predecessors {
            for &successor in &successors {
                self.set_reachable(predecessor, successor);
            }
        }
    }

    pub fn can_reach(&self, from_component: usize, to_component: usize) -> bool {
        self.reachability[from_component * self.component_count + to_component]
    }

    fn set_reachable(&mut self, from_component: usize, to_component: usize) {
        self.reachability[from_component * self.component_count + to_component] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_merge(mut merge_components: Vec<usize>) -> Vec<usize> {
        merge_components.sort_unstable();
        merge_components
    }

    #[test]
    fn keeps_acyclic_edges_between_separate_components() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();
        let c = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        assert_eq!(scc.add_edge(b, c), None);

        assert!(scc.can_reach(a, b));
        assert!(scc.can_reach(b, c));
        assert!(scc.can_reach(a, c));
        assert!(!scc.can_reach(c, a));
    }

    #[test]
    fn merges_two_components_for_bidirectional_edges() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        let merge = scc.add_edge(b, a).unwrap();

        assert_eq!(sorted_merge(merge.merged_components), vec![a, b]);
        assert_eq!(merge.component_remap, vec![0, 0]);
        assert_eq!(scc.component_count, 1);
        assert!(scc.can_reach(0, 0));
    }

    #[test]
    fn merges_cycle_path_and_preserves_external_edges() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();
        let c = scc.add_component();
        let d = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        assert_eq!(scc.add_edge(b, c), None);
        assert_eq!(scc.add_edge(c, d), None);
        let merge = scc.add_edge(c, a).unwrap();

        assert_eq!(sorted_merge(merge.merged_components), vec![a, b, c]);
        let merged = merge.component_remap[a];
        let d = merge.component_remap[d];
        assert_eq!(scc.component_count, 2);
        assert!(scc.can_reach(merged, d));
        assert!(!scc.can_reach(d, merged));
    }

    #[test]
    fn duplicate_edges_are_idempotent() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        let reachability = scc.reachability.clone();
        assert_eq!(scc.add_edge(a, b), None);

        assert_eq!(scc.reachability, reachability);
    }

    #[test]
    fn preserves_external_reachability_on_merge() {
        let mut scc = SccDag::default();
        let source = scc.add_component();
        let a = scc.add_component();
        let b = scc.add_component();
        let sink = scc.add_component();

        assert_eq!(scc.add_edge(source, a), None);
        assert_eq!(scc.add_edge(a, b), None);
        assert_eq!(scc.add_edge(b, sink), None);
        let merge = scc.add_edge(b, a).unwrap();

        assert_eq!(sorted_merge(merge.merged_components), vec![a, b]);
        let source = merge.component_remap[source];
        let merged = merge.component_remap[a];
        let sink = merge.component_remap[sink];
        assert_eq!(scc.component_count, 3);
        assert!(scc.can_reach(source, merged));
        assert!(scc.can_reach(merged, sink));
        assert!(scc.can_reach(source, sink));
        assert!(!scc.can_reach(sink, merged));
    }
}
