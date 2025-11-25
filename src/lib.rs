//! # Datalog AST - Shared Library
//!
//! Abstract Syntax Tree types for Datalog programs.
//! Used across multiple modules (M01, M04, M05) for consistency.

use std::collections::HashSet;

// ============================================================================
// Core AST Types
// ============================================================================

/// Aggregation function types for Datalog
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AggregateFunc {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

/// Arithmetic operators for expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithOp {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Sub,
    /// Multiplication (*)
    Mul,
    /// Division (/)
    Div,
    /// Modulo (%)
    Mod,
}

impl ArithOp {
    /// Parse an arithmetic operator from a string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "+" => Some(ArithOp::Add),
            "-" => Some(ArithOp::Sub),
            "*" => Some(ArithOp::Mul),
            "/" => Some(ArithOp::Div),
            "%" => Some(ArithOp::Mod),
            _ => None,
        }
    }

    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
            ArithOp::Mod => "%",
        }
    }
}

/// Arithmetic expression tree
///
/// Represents arithmetic expressions like `d + 1` or `x * y + z`.
///
/// ## Examples
///
/// ```
/// use datalog_ast::{ArithExpr, ArithOp};
///
/// // Simple: d + 1
/// let expr = ArithExpr::Binary {
///     op: ArithOp::Add,
///     left: Box::new(ArithExpr::Variable("d".to_string())),
///     right: Box::new(ArithExpr::Constant(1)),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArithExpr {
    /// A variable reference
    Variable(String),
    /// A constant value
    Constant(i64),
    /// Binary operation
    Binary {
        op: ArithOp,
        left: Box<ArithExpr>,
        right: Box<ArithExpr>,
    },
}

impl ArithExpr {
    /// Get all variables referenced in this expression
    pub fn variables(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            ArithExpr::Variable(name) => {
                vars.insert(name.clone());
            }
            ArithExpr::Constant(_) => {}
            ArithExpr::Binary { left, right, .. } => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
        }
    }

    /// Check if this is a simple variable or constant
    pub fn is_simple(&self) -> bool {
        matches!(self, ArithExpr::Variable(_) | ArithExpr::Constant(_))
    }

    /// Try to evaluate as a constant if all values are known
    pub fn try_eval_constant(&self) -> Option<i64> {
        match self {
            ArithExpr::Constant(v) => Some(*v),
            ArithExpr::Variable(_) => None,
            ArithExpr::Binary { op, left, right } => {
                let l = left.try_eval_constant()?;
                let r = right.try_eval_constant()?;
                Some(match op {
                    ArithOp::Add => l + r,
                    ArithOp::Sub => l - r,
                    ArithOp::Mul => l * r,
                    ArithOp::Div => {
                        if r == 0 { return None; }
                        l / r
                    }
                    ArithOp::Mod => {
                        if r == 0 { return None; }
                        l % r
                    }
                })
            }
        }
    }
}

impl AggregateFunc {
    /// Parse an aggregate function name
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "count" => Some(AggregateFunc::Count),
            "sum" => Some(AggregateFunc::Sum),
            "min" => Some(AggregateFunc::Min),
            "max" => Some(AggregateFunc::Max),
            "avg" => Some(AggregateFunc::Avg),
            _ => None,
        }
    }
}

/// Represents a variable or constant in Datalog
///
/// # Examples
/// ```
/// use datalog_ast::Term;
///
/// let var = Term::Variable("x".to_string());
/// let const_val = Term::Constant(42);
/// let placeholder = Term::Placeholder;  // For parser
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Variable(String),     // e.g., "x", "y", "z"
    Constant(i64),        // e.g., 42, 100
    Placeholder,          // For parser - represents "_" in Datalog
    /// Aggregation term: count<x>, sum<y>, min<z>, max<z>, avg<z>
    Aggregate(AggregateFunc, String),  // (function, variable_name)
    /// Arithmetic expression term: d + 1, x * y, etc.
    ///
    /// Used in head atoms for computed columns:
    /// ```datalog
    /// dist(y, d+1) :- dist(x, d), edge(x, y).
    /// ```
    Arithmetic(ArithExpr),
}

impl Term {
    /// Check if this term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    /// Check if this term is a constant
    pub fn is_constant(&self) -> bool {
        matches!(self, Term::Constant(_))
    }

    /// Check if this term is an aggregate
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Term::Aggregate(_, _))
    }

    /// Check if this term is an arithmetic expression
    pub fn is_arithmetic(&self) -> bool {
        matches!(self, Term::Arithmetic(_))
    }

    /// Get variable name if this is a variable
    pub fn as_variable(&self) -> Option<&str> {
        if let Term::Variable(name) = self {
            Some(name)
        } else {
            None
        }
    }

    /// Get aggregate info if this is an aggregate term
    pub fn as_aggregate(&self) -> Option<(&AggregateFunc, &str)> {
        if let Term::Aggregate(func, var) = self {
            Some((func, var))
        } else {
            None
        }
    }

    /// Get arithmetic expression if this is an arithmetic term
    pub fn as_arithmetic(&self) -> Option<&ArithExpr> {
        if let Term::Arithmetic(expr) = self {
            Some(expr)
        } else {
            None
        }
    }

    /// Get all variables referenced by this term
    pub fn variables(&self) -> std::collections::HashSet<String> {
        match self {
            Term::Variable(name) => {
                let mut set = std::collections::HashSet::new();
                set.insert(name.clone());
                set
            }
            Term::Aggregate(_, var) => {
                let mut set = std::collections::HashSet::new();
                set.insert(var.clone());
                set
            }
            Term::Arithmetic(expr) => expr.variables(),
            _ => std::collections::HashSet::new(),
        }
    }
}

/// Represents an atom like edge(x, y) or reach(x)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub relation: String,
    pub args: Vec<Term>,
}

impl Atom {
    /// Create a new atom
    pub fn new(relation: String, args: Vec<Term>) -> Self {
        Atom { relation, args }
    }

    /// Get all variables in this atom (including variables inside aggregates and arithmetic)
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for term in &self.args {
            vars.extend(term.variables());
        }
        vars
    }

    /// Check if this atom contains any aggregate terms
    pub fn has_aggregates(&self) -> bool {
        self.args.iter().any(|t| t.is_aggregate())
    }

    /// Check if this atom contains any arithmetic expressions
    pub fn has_arithmetic(&self) -> bool {
        self.args.iter().any(|t| t.is_arithmetic())
    }

    /// Get all aggregate terms in this atom
    pub fn aggregates(&self) -> Vec<(&AggregateFunc, &str)> {
        self.args
            .iter()
            .filter_map(|t| t.as_aggregate())
            .collect()
    }

    /// Get all arithmetic expressions in this atom
    pub fn arithmetic_terms(&self) -> Vec<(usize, &ArithExpr)> {
        self.args
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.as_arithmetic().map(|e| (i, e)))
            .collect()
    }

    /// Get the arity (number of arguments) of this atom
    pub fn arity(&self) -> usize {
        self.args.len()
    }
}

/// Represents a comparison constraint (x != y, x < 10, etc.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constraint {
    NotEqual(Term, Term),
    LessThan(Term, Term),
    LessOrEqual(Term, Term),
    GreaterThan(Term, Term),
    GreaterOrEqual(Term, Term),
    Equal(Term, Term),  // For completeness
}

impl Constraint {
    /// Get all variables in this constraint
    pub fn variables(&self) -> HashSet<String> {
        let (left, right) = match self {
            Constraint::NotEqual(l, r) => (l, r),
            Constraint::LessThan(l, r) => (l, r),
            Constraint::LessOrEqual(l, r) => (l, r),
            Constraint::GreaterThan(l, r) => (l, r),
            Constraint::GreaterOrEqual(l, r) => (l, r),
            Constraint::Equal(l, r) => (l, r),
        };

        let mut vars = HashSet::new();
        if let Term::Variable(name) = left {
            vars.insert(name.clone());
        }
        if let Term::Variable(name) = right {
            vars.insert(name.clone());
        }
        vars
    }
}

/// Represents a body predicate (positive or negated atom)
/// Used in rule bodies to support stratified negation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BodyPredicate {
    Positive(Atom),
    Negated(Atom),
}

impl BodyPredicate {
    /// Get the underlying atom
    pub fn atom(&self) -> &Atom {
        match self {
            BodyPredicate::Positive(atom) => atom,
            BodyPredicate::Negated(atom) => atom,
        }
    }

    /// Check if this is a positive atom
    pub fn is_positive(&self) -> bool {
        matches!(self, BodyPredicate::Positive(_))
    }

    /// Check if this is a negated atom
    pub fn is_negated(&self) -> bool {
        matches!(self, BodyPredicate::Negated(_))
    }

    /// Get all variables in this predicate
    pub fn variables(&self) -> HashSet<String> {
        self.atom().variables()
    }
}

/// Represents a single Datalog rule
///
/// # Examples
/// ```
/// // reach(y) :- reach(x), edge(x, y).
/// ```
#[derive(Debug, Clone)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<BodyPredicate>,
    pub constraints: Vec<Constraint>,
}

impl Rule {
    /// Create a new rule
    pub fn new(head: Atom, body: Vec<BodyPredicate>, constraints: Vec<Constraint>) -> Self {
        Rule {
            head,
            body,
            constraints,
        }
    }

    /// Create a rule with only positive body atoms (no negation)
    pub fn new_simple(head: Atom, body: Vec<Atom>, constraints: Vec<Constraint>) -> Self {
        Rule {
            head,
            body: body.into_iter().map(BodyPredicate::Positive).collect(),
            constraints,
        }
    }

    /// Check if this rule is safe (all head variables appear in positive body atoms)
    pub fn is_safe(&self) -> bool {
        let head_vars = self.head.variables();
        let positive_body_vars = self.positive_body_variables();

        head_vars.is_subset(&positive_body_vars)
    }

    /// Get all variables in positive body atoms
    pub fn positive_body_variables(&self) -> HashSet<String> {
        self.body
            .iter()
            .filter(|pred| pred.is_positive())
            .flat_map(|pred| pred.variables())
            .collect()
    }

    /// Get all variables in this rule
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = self.head.variables();

        for pred in &self.body {
            vars.extend(pred.variables());
        }

        for constraint in &self.constraints {
            vars.extend(constraint.variables());
        }

        vars
    }

    /// Check if this rule is recursive (head relation appears in body)
    pub fn is_recursive(&self) -> bool {
        self.body
            .iter()
            .any(|pred| pred.atom().relation == self.head.relation)
    }

    /// Get all positive body atoms
    pub fn positive_body_atoms(&self) -> Vec<&Atom> {
        self.body
            .iter()
            .filter_map(|pred| match pred {
                BodyPredicate::Positive(atom) => Some(atom),
                BodyPredicate::Negated(_) => None,
            })
            .collect()
    }

    /// Get all negated body atoms
    pub fn negated_body_atoms(&self) -> Vec<&Atom> {
        self.body
            .iter()
            .filter_map(|pred| match pred {
                BodyPredicate::Negated(atom) => Some(atom),
                BodyPredicate::Positive(_) => None,
            })
            .collect()
    }
}

/// Represents a complete Datalog program
#[derive(Debug, Clone)]
pub struct Program {
    pub rules: Vec<Rule>,
}

impl Program {
    /// Create a new empty program
    pub fn new() -> Self {
        Program { rules: Vec::new() }
    }

    /// Add a rule to the program
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Returns all IDB relations (those that appear as heads of rules)
    pub fn idbs(&self) -> Vec<String> {
        let mut idbs: Vec<String> = self
            .rules
            .iter()
            .map(|rule| rule.head.relation.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        idbs.sort();
        idbs
    }

    /// Returns all EDB relations (those that appear in bodies but never as heads)
    pub fn edbs(&self) -> Vec<String> {
        let idb_set: HashSet<String> = self.idbs().into_iter().collect();

        let mut body_relations: HashSet<String> = HashSet::new();
        for rule in &self.rules {
            for pred in &rule.body {
                body_relations.insert(pred.atom().relation.clone());
            }
        }

        let mut edbs: Vec<String> = body_relations
            .difference(&idb_set)
            .cloned()
            .collect();

        edbs.sort();
        edbs
    }

    /// Get all relation names (both EDB and IDB)
    pub fn all_relations(&self) -> Vec<String> {
        let mut all: HashSet<String> = HashSet::new();

        // Add IDBs
        for idb in self.idbs() {
            all.insert(idb);
        }

        // Add EDBs
        for edb in self.edbs() {
            all.insert(edb);
        }

        let mut result: Vec<String> = all.into_iter().collect();
        result.sort();
        result
    }

    /// Check if all rules in the program are safe
    pub fn is_safe(&self) -> bool {
        self.rules.iter().all(|rule| rule.is_safe())
    }

    /// Get all recursive rules
    pub fn recursive_rules(&self) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|rule| rule.is_recursive())
            .collect()
    }

    /// Get all non-recursive rules
    pub fn non_recursive_rules(&self) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|rule| !rule.is_recursive())
            .collect()
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let var = Term::Variable("x".to_string());
        let const_val = Term::Constant(42);
        let placeholder = Term::Placeholder;

        assert!(var.is_variable());
        assert!(const_val.is_constant());
        assert!(!placeholder.is_variable());
    }

    #[test]
    fn test_atom_creation() {
        let atom = Atom::new(
            "edge".to_string(),
            vec![Term::Variable("x".to_string()), Term::Variable("y".to_string())],
        );

        assert_eq!(atom.relation, "edge");
        assert_eq!(atom.arity(), 2);

        let vars = atom.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_rule_safety() {
        let head = Atom::new("reach".to_string(), vec![Term::Variable("y".to_string())]);

        let body = vec![
            BodyPredicate::Positive(Atom::new(
                "reach".to_string(),
                vec![Term::Variable("x".to_string())],
            )),
            BodyPredicate::Positive(Atom::new(
                "edge".to_string(),
                vec![Term::Variable("x".to_string()), Term::Variable("y".to_string())],
            )),
        ];

        let rule = Rule::new(head, body, vec![]);

        assert!(rule.is_safe());  // y appears in edge(x, y)
        assert!(rule.is_recursive());  // reach appears in head and body
    }

    #[test]
    fn test_program_edbs_idbs() {
        let mut program = Program::new();

        // reach(x) :- source(x).
        program.add_rule(Rule::new_simple(
            Atom::new("reach".to_string(), vec![Term::Variable("x".to_string())]),
            vec![Atom::new("source".to_string(), vec![Term::Variable("x".to_string())])],
            vec![],
        ));

        // reach(y) :- reach(x), edge(x, y).
        program.add_rule(Rule::new_simple(
            Atom::new("reach".to_string(), vec![Term::Variable("y".to_string())]),
            vec![
                Atom::new("reach".to_string(), vec![Term::Variable("x".to_string())]),
                Atom::new("edge".to_string(), vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                ]),
            ],
            vec![],
        ));

        let idbs = program.idbs();
        let edbs = program.edbs();

        assert_eq!(idbs, vec!["reach"]);
        assert_eq!(edbs, vec!["edge", "source"]);
    }
}
