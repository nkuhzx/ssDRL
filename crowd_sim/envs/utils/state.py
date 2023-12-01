class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta,hr_social_stress):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

        self.hr_social_stress=hr_social_stress

    def __add__(self, other):

        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,self.hr_social_stress)

    def __str__(self):

        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                              self.v_pref, self.theta,self.hr_social_stress]])
    def to_list(self):

        return (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,self.hr_social_stress)


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius,hr_social_stress=None,eye_contact=None,intention=None,id=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

        self.hr_social_stress=hr_social_stress

        self.eye_contact=eye_contact
        self.intention=intention
        self.id=id

    def __add__(self, other):

        if self.hr_social_stress is None:

            return other + (self.px, self.py, self.vx, self.vy, self.radius)

        else:

            return other + (self.px, self.py, self.vx, self.vy, self.radius,self.hr_social_stress,self.eye_contact,self.intention,self.id)

    def __str__(self):

        if self.hr_social_stress is None:

            return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

        else:

            return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius,self.hr_social_stress,self.eye_contact,self.intention,self.id]])

    def to_list(self):

        if self.hr_social_stress is None:

            return (self.px, self.py, self.vx, self.vy, self.radius)

        else:

            return (self.px, self.py, self.vx, self.vy, self.radius,self.hr_social_stress,self.eye_contact,self.intention,self.id)


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
