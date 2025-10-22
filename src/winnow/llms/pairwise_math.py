def triangular_number(n: int) -> int:
    """
    Calculate the nth triangular number.
    """
    return n * (n + 1) // 2


# This is for pairwise combinations
def triangular_number_pairwise(n: int) -> int:
    """
    Calculate the number of pairwise combinations of n items.
    """
    return triangular_number(n - 1)


if __name__ == "__main__":
    # Test the function
    for i in range(100, 2000, 100):
        print(
            f"Number of comparisons for a dataset of size {i}: {triangular_number_pairwise(i)}"
        )
