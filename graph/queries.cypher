// Parallax public showcase: three saved, parameterized graph questions.
// Schema: (:RawName)-[:RESOLVES_TO]->(:Borrower)
//         (:Lender)-[:HOLDS]->(:Borrower)
//         (:RawName)-[:SIMILAR_TO]-(:RawName)

// === Q1 co-lending network: which lenders repeatedly appear alongside a given lender
// Question (allocator / underwriter): when I lend to a name, who else is in the room, and how often?
// :param lender => "ARCC"
// :param min_shared => 1
MATCH (a:Lender {ticker: $lender})-[h1:HOLDS]->(b:Borrower)<-[h2:HOLDS]-(other:Lender)
WHERE other.ticker <> $lender AND h1.period = h2.period
WITH other, count(DISTINCT b) AS n_borrowers, count(*) AS n_borrower_periods
WHERE n_borrowers >= $min_shared
RETURN other.ticker AS lender, other.name AS lender_name, n_borrowers, n_borrower_periods
ORDER BY n_borrowers DESC, lender


// === Q2 two-hop exposure: borrowers reached from a stressed name through shared lenders
// Question (monitoring analyst): this name is in trouble; which other names sit behind the same lenders,
// :param borrower_id => "3fe4c991d7dabaa2"
// :param period => "2026-06-30"
// :param min_shared_lenders => 2
MATCH (s:Borrower {id: $borrower_id})<-[:HOLDS {period: $period}]-(l:Lender)-[h:HOLDS {period: $period}]->(b:Borrower)
WHERE b.id <> $borrower_id
WITH b, collect(DISTINCT l.ticker) AS shared_lenders,
     sum(h.cost) AS cost, sum(h.fair_value) AS fair_value,
     max(CASE WHEN h.non_accrual THEN 1 ELSE 0 END) AS na,
     collect(DISTINCT h.accession) AS accessions
WHERE size(shared_lenders) >= $min_shared_lenders
RETURN b.id AS borrower_id, b.name AS borrower, b.industry AS industry,
       shared_lenders, size(shared_lenders) AS n_shared_lenders,
       cost, fair_value,
       CASE WHEN cost > 0 THEN fair_value / cost ELSE null END AS fvc,
       na = 1 AS any_non_accrual, accessions
ORDER BY n_shared_lenders DESC, borrower


// === Q3 mark dispersion: same borrower, same period, different fair-value-to-cost ratios across filers
// Question (allocator): is my manager's mark consistent with how other lenders mark the same borrower?
// :param period => "2026-06-30"
// :param min_cost => 1000000
// :param min_dispersion => 0.05
MATCH (l:Lender)-[h:HOLDS {period: $period}]->(b:Borrower)
WHERE h.cost >= $min_cost AND h.fair_value IS NOT NULL
WITH b, collect({ticker: l.ticker, fvc: h.fair_value / h.cost, cost: h.cost,
                 non_accrual: h.non_accrual, non_accrual_debt: h.non_accrual_debt,
                 accession: h.accession}) AS holders,
     min(h.fair_value / h.cost) AS fvc_min, max(h.fair_value / h.cost) AS fvc_max,
     max(CASE WHEN h.non_accrual_debt THEN 1 ELSE 0 END) AS na
WHERE size(holders) >= 2 AND fvc_max - fvc_min >= $min_dispersion
RETURN b.id AS borrower_id, b.name AS borrower, b.industry AS industry,
       size(holders) AS n_holders, holders,
       fvc_min, fvc_max, fvc_max - fvc_min AS dispersion, na = 1 AS any_non_accrual
ORDER BY dispersion DESC, borrower
