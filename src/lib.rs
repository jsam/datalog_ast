//! # Datalog AST - Shared Library
//!
//! Abstract Syntax Tree types for Datalog programs.
//! Used across multiple modules (M01, M04, M05) for consistency.

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// Core AST Types
// ============================================================================

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
    Variable(String), // e.g., "x", "y", "z"
    Constant(i64),    // e.g., 42, 100
    Placeholder,      // For parser - represents "_" in Datalog
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

    /// Get variable name if this is a variable
    pub fn as_variable(&self) -> Option<&str> {
        if let Term::Variable(name) = self {
            Some(name)
        } else {
            None
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(name) => write!(f, "{name}"),
            Term::Constant(value) => write!(f, "{value}"),
            Term::Placeholder => write!(f, "_"),
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

    /// Get all variables in this atom
    pub fn variables(&self) -> HashSet<String> {
        self.args
            .iter()
            .filter_map(|term| {
                if let Term::Variable(name) = term {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the arity (number of arguments) of this atom
    pub fn arity(&self) -> usize {
        self.args.len()
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.relation)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{arg}")?;
        }
        write!(f, ")")
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
    Equal(Term, Term), // For completeness
}

impl Constraint {
    /// Get all variables in this constraint
    pub fn variables(&self) -> HashSet<String> {
        let (left, right) = match self {
            Constraint::NotEqual(l, r)
            | Constraint::LessThan(l, r)
            | Constraint::LessOrEqual(l, r)
            | Constraint::GreaterThan(l, r)
            | Constraint::GreaterOrEqual(l, r)
            | Constraint::Equal(l, r) => (l, r),
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

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::NotEqual(l, r) => write!(f, "{l} != {r}"),
            Constraint::LessThan(l, r) => write!(f, "{l} < {r}"),
            Constraint::LessOrEqual(l, r) => write!(f, "{l} <= {r}"),
            Constraint::GreaterThan(l, r) => write!(f, "{l} > {r}"),
            Constraint::GreaterOrEqual(l, r) => write!(f, "{l} >= {r}"),
            Constraint::Equal(l, r) => write!(f, "{l} == {r}"),
        }
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
            BodyPredicate::Positive(atom) | BodyPredicate::Negated(atom) => atom,
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

impl fmt::Display for BodyPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BodyPredicate::Positive(atom) => write!(f, "{atom}"),
            BodyPredicate::Negated(atom) => write!(f, "not {atom}"),
        }
    }
}

/// Represents a single Datalog rule
///
/// A rule consists of a head atom, a body of predicates (positive or negated),
/// and optional constraints.
///
/// # Examples
///
/// ```
/// use datalog_ast::{Atom, BodyPredicate, Rule, Term};
///
/// // reach(y) :- reach(x), edge(x, y).
/// let rule = Rule::new(
///     Atom::new("reach".into(), vec![Term::Variable("y".into())]),
///     vec![
///         BodyPredicate::Positive(Atom::new("reach".into(), vec![Term::Variable("x".into())])),
///         BodyPredicate::Positive(Atom::new("edge".into(), vec![
///             Term::Variable("x".into()),
///             Term::Variable("y".into()),
///         ])),
///     ],
///     vec![],
/// );
///
/// assert!(rule.is_safe());
/// assert!(rule.is_recursive());
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
            .flat_map(BodyPredicate::variables)
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

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.head)?;
        if self.body.is_empty() && self.constraints.is_empty() {
            write!(f, ".")
        } else {
            write!(f, " :- ")?;
            let mut first = true;
            for pred in &self.body {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{pred}")?;
                first = false;
            }
            for constraint in &self.constraints {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{constraint}")?;
                first = false;
            }
            write!(f, ".")
        }
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

        let mut edbs: Vec<String> = body_relations.difference(&idb_set).cloned().collect();

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
        self.rules.iter().all(Rule::is_safe)
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

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, rule) in self.rules.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{rule}")?;
        }
        Ok(())
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
            vec![
                Term::Variable("x".to_string()),
                Term::Variable("y".to_string()),
            ],
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
                vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                ],
            )),
        ];

        let rule = Rule::new(head, body, vec![]);

        assert!(rule.is_safe()); // y appears in edge(x, y)
        assert!(rule.is_recursive()); // reach appears in head and body
    }

    #[test]
    fn test_program_edbs_idbs() {
        let mut program = Program::new();

        // reach(x) :- source(x).
        program.add_rule(Rule::new_simple(
            Atom::new("reach".to_string(), vec![Term::Variable("x".to_string())]),
            vec![Atom::new(
                "source".to_string(),
                vec![Term::Variable("x".to_string())],
            )],
            vec![],
        ));

        // reach(y) :- reach(x), edge(x, y).
        program.add_rule(Rule::new_simple(
            Atom::new("reach".to_string(), vec![Term::Variable("y".to_string())]),
            vec![
                Atom::new("reach".to_string(), vec![Term::Variable("x".to_string())]),
                Atom::new(
                    "edge".to_string(),
                    vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("y".to_string()),
                    ],
                ),
            ],
            vec![],
        ));

        let idbs = program.idbs();
        let edbs = program.edbs();

        assert_eq!(idbs, vec!["reach"]);
        assert_eq!(edbs, vec!["edge", "source"]);
    }

    #[test]
    fn test_term_display() {
        assert_eq!(Term::Variable("x".into()).to_string(), "x");
        assert_eq!(Term::Constant(42).to_string(), "42");
        assert_eq!(Term::Placeholder.to_string(), "_");
    }

    #[test]
    fn test_term_as_variable() {
        let var = Term::Variable("foo".into());
        let constant = Term::Constant(10);

        assert_eq!(var.as_variable(), Some("foo"));
        assert_eq!(constant.as_variable(), None);
    }

    #[test]
    fn test_atom_display() {
        let atom = Atom::new(
            "edge".into(),
            vec![Term::Variable("x".into()), Term::Variable("y".into())],
        );
        assert_eq!(atom.to_string(), "edge(x, y)");

        let atom_with_constant = Atom::new("node".into(), vec![Term::Constant(1)]);
        assert_eq!(atom_with_constant.to_string(), "node(1)");

        let empty_atom = Atom::new("empty".into(), vec![]);
        assert_eq!(empty_atom.to_string(), "empty()");
    }

    #[test]
    fn test_constraint_variables() {
        let constraint =
            Constraint::NotEqual(Term::Variable("x".into()), Term::Variable("y".into()));
        let vars = constraint.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));

        let constraint_with_constant =
            Constraint::LessThan(Term::Variable("x".into()), Term::Constant(100));
        let vars = constraint_with_constant.variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_constraint_display() {
        assert_eq!(
            Constraint::NotEqual(Term::Variable("x".into()), Term::Variable("y".into()))
                .to_string(),
            "x != y"
        );
        assert_eq!(
            Constraint::LessThan(Term::Variable("x".into()), Term::Constant(10)).to_string(),
            "x < 10"
        );
        assert_eq!(
            Constraint::LessOrEqual(Term::Variable("x".into()), Term::Constant(10)).to_string(),
            "x <= 10"
        );
        assert_eq!(
            Constraint::GreaterThan(Term::Variable("x".into()), Term::Constant(0)).to_string(),
            "x > 0"
        );
        assert_eq!(
            Constraint::GreaterOrEqual(Term::Variable("x".into()), Term::Constant(0)).to_string(),
            "x >= 0"
        );
        assert_eq!(
            Constraint::Equal(Term::Variable("x".into()), Term::Variable("y".into())).to_string(),
            "x == y"
        );
    }

    #[test]
    fn test_body_predicate() {
        let atom = Atom::new("edge".into(), vec![Term::Variable("x".into())]);

        let positive = BodyPredicate::Positive(atom.clone());
        let negated = BodyPredicate::Negated(atom.clone());

        assert!(positive.is_positive());
        assert!(!positive.is_negated());
        assert!(!negated.is_positive());
        assert!(negated.is_negated());

        assert_eq!(positive.atom(), &atom);
        assert_eq!(negated.atom(), &atom);

        assert_eq!(positive.to_string(), "edge(x)");
        assert_eq!(negated.to_string(), "not edge(x)");
    }

    #[test]
    fn test_rule_display() {
        // Simple rule: reach(y) :- edge(x, y).
        let rule = Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("y".into())]),
            vec![Atom::new(
                "edge".into(),
                vec![Term::Variable("x".into()), Term::Variable("y".into())],
            )],
            vec![],
        );
        assert_eq!(rule.to_string(), "reach(y) :- edge(x, y).");

        // Rule with constraint: path(x, y) :- edge(x, y), x != y.
        let rule_with_constraint = Rule::new_simple(
            Atom::new(
                "path".into(),
                vec![Term::Variable("x".into()), Term::Variable("y".into())],
            ),
            vec![Atom::new(
                "edge".into(),
                vec![Term::Variable("x".into()), Term::Variable("y".into())],
            )],
            vec![Constraint::NotEqual(
                Term::Variable("x".into()),
                Term::Variable("y".into()),
            )],
        );
        assert_eq!(
            rule_with_constraint.to_string(),
            "path(x, y) :- edge(x, y), x != y."
        );

        // Rule with negation: safe(x) :- node(x), not dangerous(x).
        let rule_with_negation = Rule::new(
            Atom::new("safe".into(), vec![Term::Variable("x".into())]),
            vec![
                BodyPredicate::Positive(Atom::new("node".into(), vec![Term::Variable("x".into())])),
                BodyPredicate::Negated(Atom::new(
                    "dangerous".into(),
                    vec![Term::Variable("x".into())],
                )),
            ],
            vec![],
        );
        assert_eq!(
            rule_with_negation.to_string(),
            "safe(x) :- node(x), not dangerous(x)."
        );
    }

    #[test]
    fn test_rule_unsafe() {
        // Unsafe rule: result(z) :- edge(x, y). (z doesn't appear in body)
        let unsafe_rule = Rule::new_simple(
            Atom::new("result".into(), vec![Term::Variable("z".into())]),
            vec![Atom::new(
                "edge".into(),
                vec![Term::Variable("x".into()), Term::Variable("y".into())],
            )],
            vec![],
        );
        assert!(!unsafe_rule.is_safe());
    }

    #[test]
    fn test_rule_non_recursive() {
        // Non-recursive: result(x) :- source(x).
        let rule = Rule::new_simple(
            Atom::new("result".into(), vec![Term::Variable("x".into())]),
            vec![Atom::new("source".into(), vec![Term::Variable("x".into())])],
            vec![],
        );
        assert!(!rule.is_recursive());
    }

    #[test]
    fn test_rule_variables() {
        let rule = Rule::new(
            Atom::new(
                "path".into(),
                vec![Term::Variable("x".into()), Term::Variable("z".into())],
            ),
            vec![
                BodyPredicate::Positive(Atom::new(
                    "edge".into(),
                    vec![Term::Variable("x".into()), Term::Variable("y".into())],
                )),
                BodyPredicate::Positive(Atom::new(
                    "path".into(),
                    vec![Term::Variable("y".into()), Term::Variable("z".into())],
                )),
            ],
            vec![Constraint::NotEqual(
                Term::Variable("x".into()),
                Term::Variable("z".into()),
            )],
        );

        let vars = rule.variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }

    #[test]
    fn test_rule_body_atoms() {
        let rule = Rule::new(
            Atom::new("result".into(), vec![Term::Variable("x".into())]),
            vec![
                BodyPredicate::Positive(Atom::new("a".into(), vec![Term::Variable("x".into())])),
                BodyPredicate::Negated(Atom::new("b".into(), vec![Term::Variable("x".into())])),
                BodyPredicate::Positive(Atom::new("c".into(), vec![Term::Variable("x".into())])),
            ],
            vec![],
        );

        let positive = rule.positive_body_atoms();
        let negated = rule.negated_body_atoms();

        assert_eq!(positive.len(), 2);
        assert_eq!(positive[0].relation, "a");
        assert_eq!(positive[1].relation, "c");

        assert_eq!(negated.len(), 1);
        assert_eq!(negated[0].relation, "b");
    }

    #[test]
    fn test_program_display() {
        let mut program = Program::new();

        program.add_rule(Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("x".into())]),
            vec![Atom::new("source".into(), vec![Term::Variable("x".into())])],
            vec![],
        ));

        program.add_rule(Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("y".into())]),
            vec![
                Atom::new("reach".into(), vec![Term::Variable("x".into())]),
                Atom::new(
                    "edge".into(),
                    vec![Term::Variable("x".into()), Term::Variable("y".into())],
                ),
            ],
            vec![],
        ));

        let expected = "reach(x) :- source(x).\nreach(y) :- reach(x), edge(x, y).";
        assert_eq!(program.to_string(), expected);
    }

    #[test]
    fn test_program_all_relations() {
        let mut program = Program::new();

        program.add_rule(Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("x".into())]),
            vec![Atom::new("source".into(), vec![Term::Variable("x".into())])],
            vec![],
        ));

        program.add_rule(Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("y".into())]),
            vec![
                Atom::new("reach".into(), vec![Term::Variable("x".into())]),
                Atom::new(
                    "edge".into(),
                    vec![Term::Variable("x".into()), Term::Variable("y".into())],
                ),
            ],
            vec![],
        ));

        let all = program.all_relations();
        assert_eq!(all, vec!["edge", "reach", "source"]);
    }

    #[test]
    fn test_program_recursive_rules() {
        let mut program = Program::new();

        // Non-recursive
        program.add_rule(Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("x".into())]),
            vec![Atom::new("source".into(), vec![Term::Variable("x".into())])],
            vec![],
        ));

        // Recursive
        program.add_rule(Rule::new_simple(
            Atom::new("reach".into(), vec![Term::Variable("y".into())]),
            vec![
                Atom::new("reach".into(), vec![Term::Variable("x".into())]),
                Atom::new(
                    "edge".into(),
                    vec![Term::Variable("x".into()), Term::Variable("y".into())],
                ),
            ],
            vec![],
        ));

        assert_eq!(program.recursive_rules().len(), 1);
        assert_eq!(program.non_recursive_rules().len(), 1);
    }

    #[test]
    fn test_program_safety() {
        let mut safe_program = Program::new();
        safe_program.add_rule(Rule::new_simple(
            Atom::new("result".into(), vec![Term::Variable("x".into())]),
            vec![Atom::new("source".into(), vec![Term::Variable("x".into())])],
            vec![],
        ));
        assert!(safe_program.is_safe());

        let mut unsafe_program = Program::new();
        unsafe_program.add_rule(Rule::new_simple(
            Atom::new("result".into(), vec![Term::Variable("z".into())]),
            vec![Atom::new("source".into(), vec![Term::Variable("x".into())])],
            vec![],
        ));
        assert!(!unsafe_program.is_safe());
    }

    #[test]
    fn test_empty_program() {
        let program = Program::new();

        assert!(program.rules.is_empty());
        assert!(program.idbs().is_empty());
        assert!(program.edbs().is_empty());
        assert!(program.all_relations().is_empty());
        assert!(program.is_safe()); // Empty program is vacuously safe
        assert!(program.recursive_rules().is_empty());
        assert!(program.non_recursive_rules().is_empty());
        assert_eq!(program.to_string(), "");
    }

    #[test]
    fn test_program_default() {
        let program: Program = Default::default();
        assert!(program.rules.is_empty());
    }
}
