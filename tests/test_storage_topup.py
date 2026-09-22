"""
Tests for Storage Top-Up System
"""

import pytest
from dessin.economics.storage_topup import StorageTopUpManager, StorageTopUp


@pytest.fixture
def topup_manager():
    """Create a storage top-up manager"""
    return StorageTopUpManager()


def test_storage_topup_initialization(topup_manager):
    """Test storage top-up manager initializes correctly"""
    assert topup_manager.topup_history == {}
    assert topup_manager.contributors == {}


def test_topup_storage_by_owner(topup_manager):
    """Test storage top-up by model owner"""
    model_id = "model_123"
    owner = "0xowner"
    
    topup = topup_manager.topup_storage(
        model_id=model_id,
        model_owner=owner,
        payer_address=owner,
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.1,  # 1.0 GB * 100 blocks * 0.001 = 0.1
        current_expiration=1000,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xabc"
    )
    
    assert topup is not None
    assert topup.model_id == model_id
    assert topup.payer_address == owner
    assert topup.blocks_added == 100
    assert topup.new_expiration_block == 1100


def test_topup_storage_by_anyone(topup_manager):
    """Test that anyone can top-up storage, not just owner"""
    model_id = "model_123"
    owner = "0xowner"
    contributor = "0xcontributor"
    
    topup = topup_manager.topup_storage(
        model_id=model_id,
        model_owner=owner,
        payer_address=contributor,  # Different from owner
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.1,
        current_expiration=1000,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xabc"
    )
    
    assert topup is not None
    assert topup.payer_address == contributor
    assert topup.payer_address != owner


def test_insufficient_payment(topup_manager):
    """Test that insufficient payment is rejected"""
    topup = topup_manager.topup_storage(
        model_id="model_123",
        model_owner="0xowner",
        payer_address="0xpayer",
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.05,  # Only 0.05 when 0.1 required
        current_expiration=1000,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xabc"
    )
    
    assert topup is None


def test_topup_history(topup_manager):
    """Test top-up history tracking"""
    model_id = "model_123"
    
    # Make multiple top-ups
    for i in range(5):
        topup_manager.topup_storage(
            model_id=model_id,
            model_owner="0xowner",
            payer_address=f"0xpayer{i}",
            model_size_gb=1.0,
            additional_blocks=100,
            payment=0.1,
            current_expiration=1000 + (i * 100),
            storage_price_per_gb_per_block=0.001,
            tx_hash=f"0xtx{i}"
        )
    
    history = topup_manager.get_topup_history(model_id)
    assert len(history) == 5


def test_multiple_contributors(topup_manager):
    """Test tracking multiple contributors"""
    model_id = "model_123"
    
    # Owner contribution
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xowner",
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.1,
        current_expiration=1000,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx1"
    )
    
    # Contributor 1
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xcontributor1",
        model_size_gb=1.0,
        additional_blocks=50,
        payment=0.05,
        current_expiration=1100,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx2"
    )
    
    # Contributor 2
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xcontributor2",
        model_size_gb=1.0,
        additional_blocks=25,
        payment=0.025,
        current_expiration=1150,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx3"
    )
    
    contributors = topup_manager.get_contributors(model_id)
    assert len(contributors) == 3
    assert contributors["0xowner"] == 0.1
    assert contributors["0xcontributor1"] == 0.05
    assert contributors["0xcontributor2"] == 0.025


def test_total_topup_amount(topup_manager):
    """Test calculating total top-up amount"""
    model_id = "model_123"
    
    # Make multiple top-ups
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xowner",
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.1,
        current_expiration=1000,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx1"
    )
    
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xcontributor",
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.1,
        current_expiration=1100,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx2"
    )
    
    total = topup_manager.get_total_topup_amount(model_id)
    assert total == 0.2


def test_user_contribution(topup_manager):
    """Test getting specific user's contribution"""
    model_id = "model_123"
    contributor = "0xcontributor"
    
    # Make multiple contributions by same user
    for i in range(3):
        topup_manager.topup_storage(
            model_id=model_id,
            model_owner="0xowner",
            payer_address=contributor,
            model_size_gb=1.0,
            additional_blocks=100,
            payment=0.1,
            current_expiration=1000 + (i * 100),
            storage_price_per_gb_per_block=0.001,
            tx_hash=f"0xtx{i}"
        )
    
    contribution = topup_manager.get_user_contribution(model_id, contributor)
    assert contribution == pytest.approx(0.3)  # 3 * 0.1


def test_storage_statistics(topup_manager):
    """Test comprehensive storage statistics"""
    model_id = "model_123"
    
    # Owner contribution
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xowner",
        model_size_gb=1.0,
        additional_blocks=100,
        payment=0.1,
        current_expiration=1000,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx1"
    )
    
    # Contributor
    topup_manager.topup_storage(
        model_id=model_id,
        model_owner="0xowner",
        payer_address="0xcontributor",
        model_size_gb=1.0,
        additional_blocks=50,
        payment=0.05,
        current_expiration=1100,
        storage_price_per_gb_per_block=0.001,
        tx_hash="0xtx2"
    )
    
    stats = topup_manager.get_storage_statistics(model_id)
    
    assert stats["total_topups"] == 2
    assert stats["total_amount"] == pytest.approx(0.15)
    assert stats["total_blocks_added"] == 150
    assert stats["unique_contributors"] == 2
    assert "latest_topup" in stats


def test_empty_model_statistics(topup_manager):
    """Test statistics for model with no top-ups"""
    stats = topup_manager.get_storage_statistics("nonexistent_model")
    
    assert stats["total_topups"] == 0
    assert stats["total_amount"] == 0.0
    assert stats["unique_contributors"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
