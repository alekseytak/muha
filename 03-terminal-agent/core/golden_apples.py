"""Золотые яблоки — голосование роя за канонические прецеденты."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class AppleVote:
    """Голос за золотое яблоко."""
    voter_id: str
    apple_id: str
    approve: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class GoldenApple:
    """Предложение «золотого яблока» — кандидат в канон."""
    id: str
    proposal: str
    proposer: str
    votes: List[AppleVote] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    canonical: bool = False
    canonical_at: Optional[float] = None

    def vote_ratio(self) -> float:
        if not self.votes:
            return 0.0
        approvals = sum(1 for v in self.votes if v.approve)
        return approvals / len(self.votes)

    def is_canonical_by_threshold(self, threshold: float = 0.66, min_votes: int = 3) -> bool:
        return (
            len(self.votes) >= min_votes
            and self.vote_ratio() >= threshold
            and not self.canonical
        )


class GoldenApples:
    """Система предложений и голосования за золотые яблоки."""

    def __init__(self, vote_threshold: float = 0.66, min_votes: int = 3) -> None:
        self.vote_threshold = vote_threshold
        self.min_votes = min_votes
        self.apples: Dict[str, GoldenApple] = {}
        self.canonical_apples: List[GoldenApple] = []

    def propose(self, proposal: str, proposer: str = "muha-agent") -> str:
        """Предложить новое золотое яблоко."""
        apple_id = f"apple:{uuid.uuid4().hex[:8]}"
        apple = GoldenApple(
            id=apple_id,
            proposal=proposal,
            proposer=proposer,
        )
        self.apples[apple_id] = apple
        return apple_id

    def vote(self, apple_id: str, voter_id: str, approve: bool) -> bool:
        """Голосовать за яблоко."""
        if apple_id not in self.apples:
            return False
        apple = self.apples[apple_id]
        # Убираем старый голос этого пользователя если был
        apple.votes = [v for v in apple.votes if v.voter_id != voter_id]
        apple.votes.append(AppleVote(
            voter_id=voter_id,
            apple_id=apple_id,
            approve=approve,
        ))

        # Проверяем порог канонизации
        if apple.is_canonical_by_threshold(self.vote_threshold, self.min_votes):
            apple.canonical = True
            apple.canonical_at = time.time()
            self.canonical_apples.append(apple)
        return True

    def get_apple(self, apple_id: str) -> Optional[GoldenApple]:
        return self.apples.get(apple_id)

    def list_pending(self) -> List[GoldenApple]:
        """Список предложений, ожидающих голосования."""
        return [
            a for a in self.apples.values()
            if not a.canonical and len(a.votes) > 0
        ]

    def list_canonical(self) -> List[GoldenApple]:
        """Список канонических яблок."""
        return self.canonical_apples

    def stats(self) -> Dict[str, Any]:
        total = len(self.apples)
        canonical = len(self.canonical_apples)
        pending = total - canonical
        return {
            "total_proposals": total,
            "canonical": canonical,
            "pending": pending,
        }
