There can be some reasons for sudden jumps:

Because the Alpha values are frequently changing from NaN to non-NaN or vice versa. You can use backfill function to take care of this.
The other reason is that the Alpha values change rapidly from time to time. Thus decay or taking average in Alpha formula can help you in making the curve smoother.
It also may be because of too much money on one stock and if the stock value has a jump then the PNL will also have a jump in it. To tackle this you can set stock weight (Truncation) in sim settings to non-zero value between 0 and 1, preferably less than 0.1.