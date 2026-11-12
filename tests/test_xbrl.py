"""Inline XBRL fact extraction.

The synthetic fixture below reproduces EDGAR's actual iXBRL shape - scaled
`ix:nonFraction` values, `sign="-"` negatives, dated `xbrli:context` elements
and dimensioned segment contexts. The real-filing assertions live in
`test_real_filings.py`.
"""

from __future__ import annotations

import pytest

from sec_copilot.ingestion.xbrl import extract_facts, facts_to_markdown

IXBRL = """
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:xbrli="http://www.xbrl.org/2003/instance">
<body>
<div style="display:none">
  <xbrli:context id="fy2025">
    <xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0000320193</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2024-09-29</xbrli:startDate><xbrli:endDate>2025-09-27</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="fy2024">
    <xbrli:period><xbrli:startDate>2023-10-01</xbrli:startDate><xbrli:endDate>2024-09-28</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="bs2025">
    <xbrli:period><xbrli:instant>2025-09-27</xbrli:instant></xbrli:period>
  </xbrli:context>
  <xbrli:context id="seg2025">
    <xbrli:period><xbrli:startDate>2024-09-29</xbrli:startDate><xbrli:endDate>2025-09-27</xbrli:endDate></xbrli:period>
    <xbrli:segment>
      <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">aapl:AmericasSegmentMember</xbrldi:explicitMember>
    </xbrli:segment>
  </xbrli:context>
</div>
<p>Operating income was
  <ix:nonFraction name="us-gaap:OperatingIncomeLoss" contextRef="fy2025" unitRef="usd" scale="6">133,050</ix:nonFraction>
  compared with
  <ix:nonFraction name="us-gaap:OperatingIncomeLoss" contextRef="fy2024" unitRef="usd" scale="6">123,216</ix:nonFraction>.
</p>
<p>Total assets were
  <ix:nonFraction name="us-gaap:Assets" contextRef="bs2025" unitRef="usd" scale="6">359,241</ix:nonFraction>.
</p>
<p>Other expense was
  <ix:nonFraction name="us-gaap:OtherNonoperatingIncomeExpense" contextRef="fy2025" unitRef="usd" scale="6" sign="-">321</ix:nonFraction>.
</p>
<p>Americas net sales were
  <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="seg2025" unitRef="usd" scale="6">178,353</ix:nonFraction>.
</p>
<p>EPS was
  <ix:nonFraction name="us-gaap:EarningsPerShareDiluted" contextRef="fy2025" unitRef="usdPerShare" decimals="2">7.46</ix:nonFraction>.
</p>
</body></html>
"""


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "filing.htm"
    path.write_text(IXBRL, encoding="utf-8")
    return extract_facts(path)


def test_facts_are_extracted_with_concept_and_period(index):
    assert len(index) == 6
    assert index.fiscal_years == {2024, 2025}


def test_the_scale_attribute_is_applied(index):
    # scale="6" means the printed 133,050 stands for 133,050,000,000.
    fact = next(f for f in index.by_concept("OperatingIncomeLoss") if f.fiscal_year == 2025)
    assert fact.value == pytest.approx(133_050e6)
    assert fact.raw == "133,050"


def test_the_sign_attribute_makes_a_figure_negative(index):
    fact = next(iter(index.by_concept("OtherNonoperatingIncomeExpense")))
    assert fact.value == pytest.approx(-321e6)


def test_a_fiscal_year_is_taken_from_the_period_end(index):
    # The FY2025 period *begins* in 2024; the end date is what names it.
    fact = next(f for f in index.by_concept("OperatingIncomeLoss") if f.value > 130e9)
    assert fact.start == "2024-09-29"
    assert fact.end == "2025-09-27"
    assert fact.fiscal_year == 2025


def test_an_instant_context_is_recognised_as_a_balance_sheet_figure(index):
    fact = next(iter(index.by_concept("Assets")))
    assert fact.is_instant
    assert fact.end == "2025-09-27"
    assert fact.fiscal_year == 2025


def test_segment_facts_are_distinguished_from_consolidated_ones(index):
    segmented = [f for f in index.facts if not f.is_consolidated]
    assert len(segmented) == 1
    assert segmented[0].value == pytest.approx(178_353e6)
    assert any("AmericasSegmentMember" in v for _, v in segmented[0].dimensions)

    # by_concept defaults to consolidated only, which is what an analyst means
    # by "net sales" without qualification.
    assert index.by_concept("RevenueFromContract") == []
    assert index.by_concept("RevenueFromContract", consolidated_only=False)


def test_value_lookup_finds_the_period_a_figure_belongs_to(index):
    # This is the query the numeric guardrail makes.
    assert index.years_for(133_050) == {2025}
    assert index.years_for(123_216) == {2024}
    assert index.years_for(999_999) == set()


def test_value_lookup_matches_either_the_tagged_or_the_printed_scale(index):
    assert index.matching(133_050)          # as printed in the statement
    assert index.matching(133_050e6)        # as tagged
    assert index.matching(133.05e9)         # as an analyst might restate it


def test_a_non_numeric_tag_is_skipped(tmp_path):
    path = tmp_path / "f.htm"
    path.write_text(
        '<html><body><ix:nonFraction name="us-gaap:Assets" contextRef="x">&#8212;</ix:nonFraction>'
        "</body></html>",
        encoding="utf-8",
    )
    assert len(extract_facts(path)) == 0


def test_a_document_with_no_xbrl_yields_an_empty_index(tmp_path):
    path = tmp_path / "plain.htm"
    path.write_text("<html><body><p>No tagging here at all.</p></body></html>", encoding="utf-8")
    index = extract_facts(path)
    assert len(index) == 0
    assert index.years_for(1234) == set()


def test_facts_render_as_a_table(index):
    markdown = facts_to_markdown(index.facts)
    assert "| Concept | Period | Value | Unit |" in markdown
    assert "Operating Income Loss" in markdown


def test_label_humanises_the_concept_name(index):
    fact = next(iter(index.by_concept("EarningsPerShareDiluted")))
    assert fact.label == "Earnings Per Share Diluted"
    assert fact.value == pytest.approx(7.46)
