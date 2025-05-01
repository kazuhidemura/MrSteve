from gym import Wrapper


class InfoWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if hasattr(self, "is_successful"):
            info["success"] = info.pop("success", False) or self.is_successful
        return obs, reward, done, info
