from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)


str_to_gammapy_profile_dict = {
    'einasto':profiles.EinastoProfile(),
    'nfw':profiles.NFWProfile(),
    'burkert':profiles.BurkertProfile(),
    'isothermal':profiles.IsothermalProfile(),
    'moore':profiles.MooreProfile(),
}