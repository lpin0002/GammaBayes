from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)

import inspect




class DMProfiles:
    """Dark Matter Profiles"""
    NFW         = profiles.NFWProfile()
    Einasto     = profiles.EinastoProfile()
    Burkert     = profiles.BurkertProfile()
    Isothermal  = profiles.IsothermalProfile()
    Moore       = profiles.MooreProfile()


str_to_gammapy_profile_dict = {
    'einasto':DMProfiles.Einasto,
    'nfw':DMProfiles.NFW,
    'burkert':DMProfiles.Burkert,
    'isothermal':DMProfiles.Isothermal,
    'moore':DMProfiles.Moore,
}

def convert_str_profile_to_DMProfiles(profile_string):
    profile_string = profile_string.lower()

    return str_to_gammapy_profile_dict[profile_string]

def check_profile_module(profile):

    try:
        if type(profile)==str:
            profile = convert_str_profile_to_DMProfiles(profile)

        assert inspect.getmodule(profile) == profiles

        return profile

    except:
        raise Exception(f"Invalid dark matter density profile. \nPlease provide a string contained in the following list {str_to_gammapy_profile_dict.keys()} \n or a profile from the gammapy.astro.darkmatter.profiles module.")


