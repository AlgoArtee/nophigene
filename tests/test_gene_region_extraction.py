from __future__ import annotations

from typing import Any

from src.gene_region_extraction import fetch_ucsc_region, find_gene_region


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_ucsc_lookup_uses_search_endpoint_for_any_chromosome(monkeypatch) -> None:
    """UCSC lookup should not be locked to the original chr11 DRD4 prototype."""

    def fake_get(url: str, *, params: dict[str, str], timeout: int) -> FakeResponse:
        assert url == "https://api.genome.ucsc.edu/search"
        assert params == {"search": "SIRT6", "genome": "hg19"}
        assert timeout > 0
        return FakeResponse(
            {
                "positionMatches": [
                    {
                        "trackName": "knownGene",
                        "matches": [
                            {
                                "position": "chr19:4174106-4182560",
                                "posName": "SIRT6 (ENST00000337491.6)",
                            },
                            {
                                "position": "chr11:637269-640706",
                                "posName": "DRD4 (ENST00000176183.6)",
                            },
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("src.gene_region_extraction.requests.get", fake_get)

    assert fetch_ucsc_region("sirt6") == "19:4174106-4182560"


def test_ucsc_lookup_prefers_exact_symbol_matches(monkeypatch) -> None:
    """Search hits for interacting or similarly named genes should not leak in."""

    def fake_get(url: str, *, params: dict[str, str], timeout: int) -> FakeResponse:
        return FakeResponse(
            {
                "positionMatches": [
                    {
                        "trackName": "knownGene",
                        "matches": [
                            {
                                "position": "chrX:101906411-101914011",
                                "posName": "GPRASP1 (DRD4-associated sorting protein)",
                            },
                            {
                                "position": "chr11:637269-640706",
                                "posName": "DRD4 (ENST00000176183.6)",
                            },
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("src.gene_region_extraction.requests.get", fake_get)

    assert fetch_ucsc_region("DRD4") == "11:637269-640706"


def test_find_gene_region_can_use_hg38_sources(monkeypatch) -> None:
    """Build-aware lookups should switch Ensembl and UCSC to GRCh38/hg38."""
    calls: dict[str, str] = {}

    def fake_ensembl(gene_symbol: str, *, server: str) -> str:
        calls["ensembl_server"] = server
        return "15:21405401-21440499"

    def fake_ucsc(gene_symbol: str, genome: str) -> str:
        calls["ucsc_genome"] = genome
        return "15:21405401-21441499"

    monkeypatch.setattr("src.gene_region_extraction.fetch_refseq_region", lambda gene_symbol: None)
    monkeypatch.setattr("src.gene_region_extraction.fetch_ensembl_region", fake_ensembl)
    monkeypatch.setattr("src.gene_region_extraction.fetch_ucsc_region", fake_ucsc)

    result = find_gene_region("poteb3", genome_build="hg38")

    assert calls == {
        "ensembl_server": "https://rest.ensembl.org",
        "ucsc_genome": "hg38",
    }
    assert result["gene_name"] == "POTEB3"
    assert result["genome_build"] == "hg38"
    assert result["selected_region"] == "15:21405401-21441499"
    assert result["selected_sources"] == ["UCSC hg38"]
