"""Paired promotion with a multiple-comparison allowance."""

from adaptive_shop.promotion import decision

# Twenty boundaries and at most five alternatives per boundary.
ALPHA = .05 / (20 * 5)


def assess(base, dev, validation_base, validation, *, minimum, step):
    row = decision(base,dev,validation_base,validation,minimum=minimum,alpha=ALPHA,seed=step)
    row['reason'] = ('eligible' if row['promoted'] else
                     'development and validation gates failed' if not row['development']['passed']
                     and not row['heldout']['passed'] else
                     'development gate failed' if not row['development']['passed'] else 'validation gate failed')
    return row['promoted'], row


def choose_promotion(records):
    eligible = [(name,row) for name,row in records.items() if row['promoted']]
    if not eligible:
        return None
    return max(eligible,key=lambda item:(item[1]['development']['lower'],
                                        item[1]['development']['mean_utility_change']))[0]
