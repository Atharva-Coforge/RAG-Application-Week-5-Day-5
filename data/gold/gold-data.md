# Gold Evaluation Dataset

## Case 1

- `question`: How much can I spend on food each day?
- `supported`: true
- `required_answer`: Employees may claim up to $65 per day for meals while
  traveling overnight.
- `expected_section`: 1

## Case 2

- `question`: Can I book first-class airfare?
- `supported`: true
- `required_answer`: Employees must purchase economy airfare. Business-class
  airfare requires written approval from a vice president.
- `expected_section`: 3

## Case 3

- `question`: My hotel costs $250. What do I need?
- `supported`: true
- `required_answer`: Hotels are reimbursable up to $225 per night. A manager
  must approve higher rates before booking.
- `expected_section`: 2

## Case 4

- `question`: Do I need a receipt for a $20 taxi?
- `supported`: true
- `required_answer`: Receipts are required only for individual expenses of $25
  or more, so a receipt is not required for a $20 taxi expense.
- `expected_section`: 5

## Case 5

- `question`: Can I claim a limousine upgrade?
- `supported`: true
- `required_answer`: Luxury vehicle upgrades are not reimbursable.
- `expected_section`: 4

## Case 6

- `question`: Does the company reimburse gym memberships?
- `supported`: false
- `required_answer`: The provided policy does not answer this question.
- `expected_section`: null