    /// The maximum depth to recurse.
    ///
    /// The default, `None`, imposes no depth restriction.
    pub fn max_depth(&mut self, depth: Option<usize>) -> &mut WalkBuilder {
        self.max_depth = depth;
        if self.min_depth.is_some()
            && self.max_depth.is_some()
            && self.max_depth < self.min_depth
        {
            self.max_depth = self.min_depth;
        }
        self
    }
