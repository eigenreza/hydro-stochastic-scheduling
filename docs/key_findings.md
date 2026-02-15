# Key Findings: Value of Flexibility in Norwegian Hydropower Scheduling

## What this project set out to answer

How much is operational flexibility actually worth in hydropower reservoir scheduling, and where in a real cascade system does that value live. The approach combines real Norwegian hydrological and price data, Bayesian forecasting, and stochastic optimization, comparing a policy that commits to a release plan in advance against one that can react as the future unfolds, with both checked against a perfect foresight benchmark.

Two models were built. The first treats the Arendalsvassdraget watercourse as a single aggregated reservoir using a system level effective head. The second is a genuine three plant cascade, Jørundland, Evenstad, and Rygene, in their real upstream to downstream order, each with its own real capacity, head, and local inflow.

## Finding 1: Flexibility value concentrates where storage actually exists

In the cascade model, Evenstad and Rygene contribute exactly zero value of flexibility, in every single year of the 22 year backtest, regardless of which routing assumption is used between plants. This is not a data gap. It can be shown directly from the numbers: the local inflow arriving at each of these two plants exceeds their turbine capacity in essentially every week of the entire 26 year record. Both plants are running flat out all the time no matter what policy is used, so there is no decision left to make and nothing for flexibility to be worth.

All of the measurable flexibility value in the cascade sits at Jørundland, the one plant with meaningful reservoir storage. This matches intuition once you see it, but it is a genuinely useful result: in a real cascade, looking for value of flexibility at a run of river plant is the wrong place to look, no matter how sophisticated the optimization is. Storage is what creates the opportunity to wait, and only nodes with storage can capture it.

## Finding 2: There is a real, small, detectable flexibility premium in normal market years

Restricting to years without an extreme price shock, Jørundland shows a positive value of flexibility of around 3.4 million NOK per year, small relative to total revenue, but consistent in sign across two different routing assumptions and statistically significant. This is the cleanest version of the project's original question, and the answer in ordinary market conditions is yes, being able to react is worth something, it is just modest.

## Finding 3: Energy crisis years break naive flexibility estimates, and we know why

The 22 year sample includes the 2021 to 2022 European energy crisis, when Norwegian electricity prices spiked far outside anything in the historical record the forecasting models were trained on. In those years, the value of flexibility numbers become wild and inconsistent, sometimes large and positive, sometimes large and negative, and the sign even flips depending on a one week routing delay assumption that barely matters in any normal year.

This was tracked down to a specific, checkable cause rather than left as unexplained noise. The week by week shape of how prices moved during 2022 was effectively the opposite of the historical seasonal pattern the model had learned, summer and winter roles were reversed. Once the underlying scenario forecasts no longer resemble reality, both the commit in advance policy and the react as you go policy are working from the same broken map, and whichever one happens to be luckier about timing in that particular year comes out ahead by chance rather than by skill. The size of that randomness scales directly with how extreme the price shock is.

The same pattern shows up independently in the single reservoir model from earlier in the project, at a larger scale that matches the difference in reservoir size between the two models almost exactly. That consistency is reassuring. It means this is a real, structural feature of how these models behave under regime shifts, not a quirk of one particular setup.

## What this adds up to

Put together, the honest version of the result is this. Flexibility in hydropower scheduling has real, measurable value, but only where physical storage exists, and that value is modest in normal conditions. When markets move into territory the forecasting model has never seen before, naive estimates of flexibility value become unreliable, not because the optimization is wrong, but because the forecasts feeding it are no longer describing the world that actually happened. Recognising that distinction, and being able to show precisely where it comes from, is arguably a more useful contribution than a single clean headline number would have been.

## What is still open

The cascade currently uses a simplified routing assumption between plants rather than precise hydraulic travel time data, which was not available from public sources. The closed loop policy is implemented as a tractable multistage approximation rather than full stochastic dual dynamic programming, the established method in this literature for medium term hydropower scheduling. Both are documented limitations, not hidden ones, and both are reasonable next steps if this work continues.
