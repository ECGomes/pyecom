# Based on the paper found in
# https://onlinelibrary.wiley.com/doi/abs/10.1049/iet-gtd.2015.0423

# The following is the bibtex entry for the paper
"""
 @article{tenfen_lithium-ion_2016,
    title = {Lithium-ion battery modelling for the energy management problem of microgrids},
    volume = {10},
    issn = {1751-8695},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1049/iet-gtd.2015.0423},
    doi = {10.1049/iet-gtd.2015.0423},
    pages = {576--584},
    number = {3},
    journaltitle = {{IET} Generation, Transmission \& Distribution},
    author = {Tenfen, Daniel and Finardi, Erlon C. and Delinchant, Benoit and Wurtz, Frédéric},
    date = {2016},
    langid = {english}}
"""


def battery_capital_costs(capital_cost, maximum_capacity,
                          soc_target, quadratic_charge_cost,
                          current_soc,
                          p_charge, p_discharge,
                          efficiency_charge, efficiency_discharge):
    return capital_cost * (current_soc / maximum_capacity - soc_target) ** 2 + \
        p_discharge * efficiency_discharge + p_charge * efficiency_charge + \
        quadratic_charge_cost / maximum_capacity * p_charge ** 2
