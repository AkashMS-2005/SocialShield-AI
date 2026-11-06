"""
adapters/platforms.py – Platform-specific adapters mapping idiosyncratic column aliases.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from app.adapters.base import BasePlatformAdapter
from app.schemas import AccountInput, Platform


def _get_first(data: Dict[str, Any], keys: list[str]) -> Any:
    """Helper to return the first matching non-null key from a dict."""
    for k in keys:
        if k in data and data[k] is not None and str(data[k]).strip() != "":
            return data[k]
    return None


def _to_int(val: Any) -> Optional[int]:
    if val is None or str(val).strip() == "":
        return None
    try:
        f = float(val)
        return int(f)
    except (ValueError, TypeError):
        return None


def _to_bool(val: Any) -> Optional[bool]:
    if val is None or str(val).strip() == "":
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ["1", "true", "yes", "y", "t"]:
        return True
    if s in ["0", "false", "no", "n", "f"]:
        return False
    return None


class InstagramAdapter(BasePlatformAdapter):
    @property
    def platform(self) -> Platform:
        return Platform.instagram

    def normalize_input(self, raw_data: Dict[str, Any]) -> AccountInput:
        username = _get_first(raw_data, ["username", "handle", "user", "profile_name"]) or "unknown_user"
        followers = _to_int(_get_first(raw_data, ["followers", "#followers", "follower_count", "flr", "edge_followed_by"]))
        following = _to_int(_get_first(raw_data, ["following", "#follows", "follows", "following_count", "flg", "edge_follow"]))
        posts = _to_int(_get_first(raw_data, ["posts", "#posts", "media_count", "pos", "post_count"]))
        age = _to_int(_get_first(raw_data, ["account_age_days", "age_days", "age", "days_active"]))
        has_pic = _to_bool(_get_first(raw_data, ["has_profile_pic", "profile pic", "profile_pic", "has_avatar"]))
        bio = _get_first(raw_data, ["bio", "description", "biography", "desc"])
        post_text = _get_first(raw_data, ["post_text", "caption", "text", "recent_posts"])

        return AccountInput(
            platform=self.platform,
            username=str(username).strip(),
            followers=followers,
            following=following,
            posts=posts,
            account_age_days=age,
            has_profile_pic=has_pic,
            bio=str(bio).strip() if bio else None,
            post_text=str(post_text).strip() if post_text else None,
        )


class XAdapter(BasePlatformAdapter):
    @property
    def platform(self) -> Platform:
        return Platform.x

    def normalize_input(self, raw_data: Dict[str, Any]) -> AccountInput:
        username = _get_first(raw_data, ["username", "screen_name", "handle", "user"]) or "unknown_user"
        followers = _to_int(_get_first(raw_data, ["followers", "followers_count", "follower_count"]))
        following = _to_int(_get_first(raw_data, ["following", "friends_count", "following_count"]))
        posts = _to_int(_get_first(raw_data, ["posts", "statuses_count", "tweet_count", "tweets"]))
        age = _to_int(_get_first(raw_data, ["account_age_days", "age_days", "days_since_created"]))
        has_pic = _to_bool(_get_first(raw_data, ["has_profile_pic", "default_profile_image", "has_avatar"]))
        if "default_profile_image" in raw_data:
            # Note: default_profile_image=True means NO custom pic
            def_img = _to_bool(raw_data["default_profile_image"])
            if def_img is not None:
                has_pic = not def_img

        bio = _get_first(raw_data, ["bio", "description"])
        post_text = _get_first(raw_data, ["post_text", "tweets", "text", "recent_tweets"])

        return AccountInput(
            platform=self.platform,
            username=str(username).strip().lstrip("@"),
            followers=followers,
            following=following,
            posts=posts,
            account_age_days=age,
            has_profile_pic=has_pic,
            bio=str(bio).strip() if bio else None,
            post_text=str(post_text).strip() if post_text else None,
        )


class YouTubeAdapter(BasePlatformAdapter):
    @property
    def platform(self) -> Platform:
        return Platform.youtube

    def normalize_input(self, raw_data: Dict[str, Any]) -> AccountInput:
        # YouTube uses 'subscribers' for followers and 'videos' for posts
        username = _get_first(raw_data, ["username", "channel_title", "channel_name", "channel", "user"]) or "unknown_channel"
        followers = _to_int(_get_first(raw_data, ["subscribers", "subscriber_count", "followers"]))
        # YouTube channels rarely have public 'following', map if present
        following = _to_int(_get_first(raw_data, ["following", "subscriptions", "subscription_count"]))
        posts = _to_int(_get_first(raw_data, ["video_count", "videos", "posts"]))
        age = _to_int(_get_first(raw_data, ["account_age_days", "channel_age_days", "days_active"]))
        has_pic = _to_bool(_get_first(raw_data, ["has_profile_pic", "has_avatar", "has_thumbnail"]))
        bio = _get_first(raw_data, ["bio", "channel_description", "description", "about"])
        post_text = _get_first(raw_data, ["post_text", "video_descriptions", "comments", "text"])

        return AccountInput(
            platform=self.platform,
            username=str(username).strip(),
            followers=followers,
            following=following,
            posts=posts,
            account_age_days=age,
            has_profile_pic=has_pic,
            bio=str(bio).strip() if bio else None,
            post_text=str(post_text).strip() if post_text else None,
        )


class FacebookAdapter(BasePlatformAdapter):
    @property
    def platform(self) -> Platform:
        return Platform.facebook

    def normalize_input(self, raw_data: Dict[str, Any]) -> AccountInput:
        username = _get_first(raw_data, ["username", "page_name", "profile_name", "user"]) or "unknown_profile"
        followers = _to_int(_get_first(raw_data, ["followers", "friends", "follower_count", "friends_count", "page_likes"]))
        following = _to_int(_get_first(raw_data, ["following", "following_count"]))
        posts = _to_int(_get_first(raw_data, ["posts", "post_count"]))
        age = _to_int(_get_first(raw_data, ["account_age_days", "age_days"]))
        has_pic = _to_bool(_get_first(raw_data, ["has_profile_pic", "has_avatar"]))
        bio = _get_first(raw_data, ["bio", "about", "intro", "description"])
        post_text = _get_first(raw_data, ["post_text", "posts_content", "text"])

        return AccountInput(
            platform=self.platform,
            username=str(username).strip(),
            followers=followers,
            following=following,
            posts=posts,
            account_age_days=age,
            has_profile_pic=has_pic,
            bio=str(bio).strip() if bio else None,
            post_text=str(post_text).strip() if post_text else None,
        )


# Registry of adapters
ADAPTER_MAP: Dict[Platform, BasePlatformAdapter] = {
    Platform.instagram: InstagramAdapter(),
    Platform.x: XAdapter(),
    Platform.youtube: YouTubeAdapter(),
    Platform.facebook: FacebookAdapter(),
}


def get_adapter_for_platform(platform: Union[Platform, str]) -> BasePlatformAdapter:
    """Finds the adapter for a given Platform enum or string."""
    if isinstance(platform, str):
        # Case-insensitive resolution
        p_clean = platform.strip().lower()
        mapping = {
            "instagram": Platform.instagram,
            "x": Platform.x,
            "twitter": Platform.x,
            "youtube": Platform.youtube,
            "facebook": Platform.facebook,
        }
        if p_clean not in mapping:
            raise ValueError(f"Unsupported social platform '{platform}'. Supported: Instagram, Facebook, X, YouTube.")
        plat_enum = mapping[p_clean]
    else:
        plat_enum = platform

    if plat_enum not in ADAPTER_MAP:
        raise ValueError(f"No adapter registered for platform {plat_enum}")
    return ADAPTER_MAP[plat_enum]

