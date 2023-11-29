import numpy as np
import math

from matplotlib import pyplot as plt
STOPPING=0
WAITTING=1
WAITTING_TO_WALKING=2
WALKING=3

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    # if the segment recede to point ,calucate the distance between the two point
    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def Ellipse(x0,y0,a,b,start=-np.pi/2,stop=np.pi/2,rotate_angle=0):
    theta=np.arange(start,stop,0.01)
    x=x0+a*np.cos(theta)*math.cos(rotate_angle)-b*np.sin(theta)*math.sin(rotate_angle)
    y = y0 + a * np.cos(theta) * math.sin(rotate_angle) + b * np.sin(theta) * math.cos(rotate_angle)

    return(x,y)


def read_list(str):
    str = str.replace('[', '')
    str = str.replace(']', '')
    list_out=map(lambda i: i, str.split(','))
    list_out=list(list_out)
    return list_out

###basic analytic geometry class###

###point
##method:
#print control
#vector:pt-pt1
#distance:d
class Point(object):
    def __init__(self, xParam=0.0, yParam=0.0):
        self.x = xParam
        self.y = yParam

    def __str__(self):
        return "Point (%f, %f)" % (self.x, self.y)

    def __sub__(self,pt):
        vector_a=self.x-pt.x
        vector_b=self.y-pt.y
        return Vector(vector_a,vector_b)

    def distance(self, pt):
        xDiff = self.x - pt.x
        yDiff = self.y - pt.y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def sum(self, pt):
        newPt = Point()
        xNew = self.x + pt.x
        yNew = self.y + pt.y
        return Point(xNew, yNew)


###Vector
##method:
#print control
#+:v1+v2
#-:v1-v1
#dot mul:v1*v2
#norm
class Vector(object):
    def __init__(self, a=0.0, b=0.0):
        self.a = a
        self.b = b

    def __str__(self):
        return 'Vector (%f, %f)' % (self.a, self.b)

    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)

    def __sub__(self,other):
        return Vector(self.a - other.a, self.b - other.b)

    def __mul__(self,other):
        return self.a * other.a+ self.b * other.b

    def norm(self):
        result = math.sqrt(self.a**2 + self.b**2)
        return result

###Line
##method:
#line gradient:k
#Point_OnSegEment:bool
#Vertical_Point:(vx,vy)
class Line(object):
    def __init__(self, pt1=Point(0,0), pt2=Point(0.0)):
        self.pt1 = pt1
        self.pt2 = pt2

    def gradient(self):
        if self.pt2.x ==self.pt1.x:
            k=float('inf')
        else:
            k = (self.pt2.y - self.pt1.y) /(self.pt2.x - self.pt1.x)
        return k

    def Point_OnSegEment(self,other):
        v1=other-self.pt1
        v2=other-self.pt2
        v=v1*v2
        if v<=0:
            return True
        else:
            return False

    def PointPro_OnSegement(self,other):
        v1=other-self.pt1
        v2=self.pt2-self.pt1
        v3=other-self.pt2
        v4=self.pt1-self.pt2
        if v1*v2>=0 and v3*v4>=0:
            return True
        else:
            return False

    def Vertical_Point(self,other):
        v1=other-self.pt1
        v2=self.pt2-self.pt1
        v3=Vector()
        dis=v1*v2/v2.norm()
        v3.a,v3.b=(dis/v2.norm())*v2.a,(dis/v2.norm())*v2.b
        v4=Vector(self.pt1.x,self.pt1.y)
        temp=v3+v4

        return Point(temp.a,temp.b)


#求机器人和人连线在机器人上的切点
def tanget_point(ptE0, Ea, Eb, Eangle, line):
    CEangle = math.cos(math.radians(Eangle))
    SEangle = math.sin(math.radians(Eangle))
    TEangle = math.tan(math.radians(Eangle))
    line_k = line.gradient()
    if line_k == 0:
        t = math.atan(-(Eb / Ea) * TEangle)
        xt = Ea * math.cos(t) * CEangle - Eb * math.sin(t) * SEangle + ptE0.x
        yt = Ea * math.cos(t) * SEangle + Eb * math.sin(t) * CEangle + ptE0.y
        xt1 = 2 * ptE0.x - xt
        yt1 = 2 * ptE0.y - yt
        if line.Point_OnSegEment(Point(xt, yt)):
            return Point(xt, yt)
        elif line.Point_OnSegEment(Point(xt1, yt1)):
            return Point(xt1, yt1)
    else:
        kt = -1 / line_k
        CEangle = math.cos(math.radians(Eangle))
        SEangle = math.sin(math.radians(Eangle))

        com = math.sqrt((pow(Eb, 2) * pow(kt, 2) + pow(Ea, 2)) * pow(SEangle, 2) + 2 * SEangle * CEangle * kt * (
                    pow(Eb, 2) - pow(Ea, 2)) + pow(CEangle, 2) * (pow(Ea, 2) * pow(kt, 2) + pow(Eb, 2)))
        deltax = -(pow(Eb, 2) * kt * pow(SEangle, 2) + (pow(Eb, 2) - pow(Ea, 2)) * CEangle * SEangle + pow(Ea, 2) * pow(
            CEangle, 2) * kt) / com
        deltay = (pow(Ea, 2) * pow(SEangle, 2) + SEangle * CEangle * kt * (pow(Eb, 2) - pow(Ea, 2)) + pow(Eb, 2) * pow(
            CEangle, 2)) / com
        xt = deltax + ptE0.x
        yt = deltay + ptE0.y
        xt1 = -deltax + ptE0.x
        yt1 = -deltay + ptE0.y

        if line.PointPro_OnSegement(Point(xt, yt)):
            return Point(xt, yt)
        elif line.PointPro_OnSegement(Point(xt1, yt1)):
            return Point(xt1, yt1)


#参数修改公式
def new_Ellipse_para(ptS, ptE0,Eangle, kpara):
    CEangle = math.cos(math.radians(Eangle))
    SEangle = math.sin(math.radians(Eangle))

    a=b=ptS.distance(ptE0)

    return float(a), float(b)

#求解人机是否相互挤压以及相互挤压的空间方向
def hr_intersection_area(human_,robot_,shapepara):

    #计算可能挤压的方向并提取用来计算的椭圆参数
    vhptorp = robot_.pos - human_.pos
    vhptoa = Vector(human_.afront * math.cos(math.radians(human_.angle)),
                    human_.afront * math.sin(math.radians(human_.angle)))
    vhptob=Vector(-human_.bleft*math.sin(math.radians(human_.angle)),human_.bleft*math.cos(math.radians(human_.angle)))
    if vhptorp * vhptoa>0:
        human_intersection_area='front'
        area_num=0
        Eha=human_.afront
    else:
        human_intersection_area='back'
        area_num=2
        Eha=human_.aback
    if vhptorp*vhptob>0:
        human_intersection_area1='left'
        area1_num=0
        Ehb=human_.bleft
    else:
        human_intersection_area1 = 'right'
        area1_num=1
        Ehb = human_.bright

    area=area_num+area1_num+1

    vrptohp = human_.pos -robot_.pos
    vrptoa = Vector(robot_.afront * math.cos(math.radians(robot_.angle)),
                    robot_.afront * math.sin(math.radians(robot_.angle)))
    if vrptohp * vrptoa>0:
        robot_intersection_area='front'
        Era=robot_.afront
        Erb=robot_.radius
    else:
        robot_intersection_area='back'
        Era=robot_.radius
        Erb=robot_.radius



    if human_.pos.distance(robot_.pos)<(human_.afront+robot_.afront):
    #print(human_intersection_area,human_intersection_area1,robot_intersection_area)
    #计算垂直于人机连线的人和机器人切点
        line_hr_center = Line(human_.pos, robot_.pos)

        tangentr = tanget_point(robot_.pos, Era, Erb, robot_.angle, line_hr_center)

        try:
            tangentr_v = line_hr_center.Vertical_Point(tangentr)
        except:
            test_collesion=collision_detected(human_,robot_)
            print(test_collesion)
            print(human_.pos,human_.afront,human_.aback,human_.bleft,human_.bright,human_.angle)
            print(robot_.pos, robot_.afront, robot_.radius, robot_.angle)

        tangenth = tanget_point(human_.pos, Eha, Ehb, human_.angle, line_hr_center)
        if tangenth==None:
            human_.squeeze_area=area
        else:
            tangenth_v = line_hr_center.Vertical_Point(tangenth)
            disr_trv=robot_.pos.distance(tangentr_v)
            dish_thv=human_.pos.distance(tangenth_v)
            disr_h=robot_.pos.distance(human_.pos)
            if disr_trv+dish_thv>disr_h:
                human_.squeeze_area=area
            else:
                human_.squeeze_area=False

        #计算是否挤压并求人的空间参数
        if human_.squeeze_area:
            changea,changeb=new_Ellipse_para(tangentr_v, human_.pos, human_.angle,shapepara)
            if area == 1:
                human_.afront, human_.bleft = changea if changea<human_.afront else human_.afront,changeb if changeb<human_.bleft else human_.bleft
            elif area == 3:
                human_.aback, human_.bleft = changea if changea<human_.aback else human_.aback,changeb if changeb<human_.bleft else human_.bleft
            elif area == 4:
                human_.aback, human_.bright = changea if changea<human_.aback else human_.aback,changeb if changeb<human_.bright else human_.bright
            elif area == 2:
                human_.afront, human_.bright =changea if changea<human_.afront else human_.afront,changeb if changeb<human_.bright else human_.bright


def collision_detected(human_,robot_):
    #计算可能挤压的方向并提取用来计算的椭圆参数
    #print(robot_.afront)
    vrptohp = human_.pos -robot_.pos
    vrptoa = Vector(robot_.afront * math.cos(math.radians(robot_.angle)),
                    robot_.afront * math.sin(math.radians(robot_.angle)))
    if vrptohp * vrptoa>0:
        Era=robot_.afront
        Erb=robot_.radius
    else:
        Era=robot_.radius
        Erb=robot_.radius


    if human_.pos.distance(robot_.pos)<(human_.afront+robot_.afront):
    #print(human_intersection_area,human_intersection_area1,robot_intersection_area)
    #计算垂直于人机连线的人和机器人切点

        line_hr_center = Line(human_.pos, robot_.pos)

        tangentr = tanget_point(robot_.pos, Era, Erb, robot_.angle, line_hr_center)

        if tangentr==None:
            collision=True
        else:
            tangentr_v = line_hr_center.Vertical_Point(tangentr)
            if tangentr_v.distance(human_.pos)<human_.radius:
                collision=True
            else:
                collision=False

    else:
        collision=False

    return collision


def hr_intersection_area_backup(human_, robot_, shapepara):
    # 计算可能挤压的方向并提取用来计算的椭圆参数
    collision=False
    vhptorp = robot_.pos - human_.pos
    vhptoa = Vector(human_.afront * math.cos(math.radians(human_.angle)),
                    human_.afront * math.sin(math.radians(human_.angle)))
    vhptob = Vector(-human_.bleft * math.sin(math.radians(human_.angle)),
                    human_.bleft * math.cos(math.radians(human_.angle)))
    if vhptorp * vhptoa > 0:
        human_intersection_area = 'front'
        area_num = 0
        Eha = human_.afront
    else:
        human_intersection_area = 'back'
        area_num = 2
        Eha = human_.aback
    if vhptorp * vhptob > 0:
        human_intersection_area1 = 'left'
        area1_num = 0
        Ehb = human_.bleft
    else:
        human_intersection_area1 = 'right'
        area1_num = 1
        Ehb = human_.bright

    area = area_num + area1_num + 1

    vrptohp = human_.pos - robot_.pos
    vrptoa = Vector(robot_.afront * math.cos(math.radians(robot_.angle)),
                    robot_.afront * math.sin(math.radians(robot_.angle)))
    if vrptohp * vrptoa > 0:
        robot_intersection_area = 'front'
        Era = robot_.afront
        Erb = robot_.radius
    else:
        robot_intersection_area = 'back'
        Era = robot_.radius
        Erb = robot_.radius

    maxpara=max(human_.afront,human_.aback,human_.bleft,human_.bright)
    if human_.pos.distance(robot_.pos) < (maxpara + robot_.afront):
        # print(human_intersection_area,human_intersection_area1,robot_intersection_area)
        # 计算垂直于人机连线的人和机器人切点
        line_hr_center = Line(human_.pos, robot_.pos)

        tangentr = tanget_point(robot_.pos, Era, Erb, robot_.angle, line_hr_center)

        if tangentr==None:
            collision=True
        else:
            tangentr_v = line_hr_center.Vertical_Point(tangentr)
            if tangentr_v.distance(human_.pos)<human_.radius:
                collision=True
            else:
                '''
        
                try:
                    tangentr_v = line_hr_center.Vertical_Point(tangentr)
                except:
                    test_collesion = collision_detected(human_, robot_)
                    print(test_collesion)
                    print(human_.pos, human_.afront, human_.aback, human_.bleft, human_.bright, human_.angle)
                    print(robot_.pos, robot_.afront, robot_.radius, robot_.angle)
                    
                '''

                tangenth = tanget_point(human_.pos, Eha, Ehb, human_.angle, line_hr_center)
                if tangenth == None:
                    human_.squeeze_area = area
                else:
                    tangenth_v = line_hr_center.Vertical_Point(tangenth)
                    disr_trv = robot_.pos.distance(tangentr_v)
                    dish_thv = human_.pos.distance(tangenth_v)
                    disr_h = robot_.pos.distance(human_.pos)
                    if disr_trv + dish_thv > disr_h:
                        human_.squeeze_area = area
                    else:
                        human_.squeeze_area = False

                # 计算是否挤压并求人的空间参数
                if human_.squeeze_area:
                    changea, changeb = new_Ellipse_para(tangentr_v, human_.pos, human_.angle, shapepara)
                    if area == 1:
                        human_.afront, human_.bleft = changea if changea < human_.afront else human_.afront, changeb if changeb < human_.bleft else human_.bleft
                    elif area == 3:
                        human_.aback, human_.bleft = changea if changea < human_.aback else human_.aback, changeb if changeb < human_.bleft else human_.bleft
                    elif area == 4:
                        human_.aback, human_.bright = changea if changea < human_.aback else human_.aback, changeb if changeb < human_.bright else human_.bright
                    elif area == 2:
                        human_.afront, human_.bright = changea if changea < human_.afront else human_.afront, changeb if changeb < human_.bright else human_.bright

                if human_.afront<human_.radius or human_.aback<human_.radius or human_.bleft<human_.radius or human_.bright<human_.radius:
                    collision=True
                else:
                    collision=False

    else:
        human_.squeeze_area=False

    return collision


def hh_intersection_area(human_,human1_,shapepara):

    #计算可能挤压的方向并提取用来计算的椭圆参数
    vhptoh1p=human1_.pos-human_.pos
    vhptoa = Vector(human_.afront * math.cos(math.radians(human_.angle)),
                    human_.afront * math.sin(math.radians(human_.angle)))
    vhptob=Vector(-human_.bleft*math.sin(math.radians(human_.angle)),human_.bleft*math.cos(math.radians(human_.angle)))

    vh1ptohp=human_.pos-human1_.pos
    vh1ptoa = Vector(human1_.afront * math.cos(math.radians(human1_.angle)),
                    human1_.afront * math.sin(math.radians(human1_.angle)))
    vh1ptob=Vector(-human1_.bleft*math.sin(math.radians(human1_.angle)),human1_.bleft*math.cos(math.radians(human1_.angle)))

    if vhptoh1p*vhptoa>0 :
        h_area=0
        Eha=human_.afront
    else:
        h_area=2
        Eha=human_.aback
    if vhptoh1p*vhptob>0:
        h_area=h_area+0
        Ehb=human_.bleft
    else:
        h_area=h_area+1
        Ehb=human_.bright

    if vh1ptohp*vh1ptoa>0:
        h1_area=0
        Eh1a=human1_.afront
    else:
        h1_area=2
        Eh1a=human1_.aback
    if vh1ptohp*vh1ptob>0:
        h1_area=h1_area+0
        Eh1b=human1_.bleft
    else:
        h1_area=h1_area+1
        Eh1b=human1_.bright

    squeeze=False
    maxpara=max(human_.afront,human_.aback,human_.bleft,human_.bright)
    maxpara1=max(human1_.afront,human1_.aback,human1_.bleft,human1_.bright)
    if human_.pos.distance(human1_.pos)<(maxpara+maxpara1):

        line_hh_center = Line(human_.pos, human1_.pos)
        tangent_h=tanget_point(human_.pos,Eha,Ehb,human_.angle,line_hh_center)
        #vertical_tangent_h=line_hh_center.Vertical_Point(tangent_h)

        tangent_h1=tanget_point(human1_.pos,Eh1a,Eh1b,human1_.angle,line_hh_center)
        #vertical_tangent_h1=line_hh_center.Vertical_Point(tangent_h1)
        if tangent_h1==None or tangent_h==None:
            squeeze=True
        else:
            vertical_tangent_h = line_hh_center.Vertical_Point(tangent_h)
            vertical_tangent_h1 = line_hh_center.Vertical_Point(tangent_h1)
            dish_proh=human_.pos.distance(vertical_tangent_h)
            dish1_proh1=human1_.pos.distance(vertical_tangent_h1)
            if dish1_proh1+dish_proh>human_.pos.distance(human1_.pos):
                squeeze=True
            else:
                squeeze=False
    else:
        squeeze=False


    #print(h_area,h1_area,squeeze)

    if squeeze:

        if human_.intention==WAITTING:
            human_velocity=0.8*human_.v_pref_backup
        elif human_.intention==WAITTING_TO_WALKING:
            human_velocity=human_.v_pref_backup
        else:
            human_velocity=math.sqrt(math.pow(human_.vx, 2) + math.pow(human_.vy, 2))

        if human1_.intention==WAITTING:
            human1_velocity = 0.8*human1_.v_pref_backup
        elif human1_.intention==WAITTING_TO_WALKING:
            human1_velocity = human1_.v_pref_backup
        else:
            human1_velocity = math.sqrt(math.pow(human1_.vx, 2) + math.pow(human1_.vy, 2))

        hh1_dis=human_.pos.distance(human1_.pos)
        v_center=math.sqrt((pow(human_velocity,2)+pow(human1_velocity,2))/2)

        if human_velocity==human1_velocity:
            Split_dis=0.5*hh1_dis
        else:
            Split_dis=hh1_dis*abs(human1_velocity-v_center)/abs(human_velocity-human1_velocity)

        if Split_dis<human_.radius:
            Split_dis=human_.radius
        if (hh1_dis-Split_dis)<human1_.radius:
            Split_dis=hh1_dis-human1_.radius


        v_htoS=Vector((Split_dis/vhptoh1p.norm())*vhptoh1p.a,(Split_dis/vhptoh1p.norm())*vhptoh1p.b)
        Split=Point((v_htoS+Vector(human_.pos.x,human_.pos.y)).a,(v_htoS+Vector(human_.pos.x,human_.pos.y)).b)

        h_changea,h_changeb=new_Ellipse_para(Split,human_.pos,human_.angle,shapepara)
        if h_area==0:
            human_.afront,human_.bleft=h_changea if h_changea<human_.afront else human_.afront,h_changeb if h_changeb<human_.bleft else human_.bleft
        elif h_area==1:
            human_.afront,human_.bright = h_changea if h_changea < human_.afront else human_.afront, h_changeb if h_changeb < human_.bright else human_.bright
        elif h_area==2:
            human_.aback,human_.bleft = h_changea if h_changea < human_.aback else human_.aback, h_changeb if h_changeb < human_.bleft else human_.bleft
        elif h_area==3:
            human_.aback, human_.bright = h_changea if h_changea < human_.aback else human_.aback, h_changeb if h_changeb < human_.bright else human_.bright

        h1_changea,h1_changeb=new_Ellipse_para(Split,human1_.pos,human1_.angle,shapepara)
        if h1_area==0:
            human1_.afront,human1_.bleft=h1_changea if h1_changea<human1_.afront else human1_.afront,h1_changeb if h1_changeb<human1_.bleft else human1_.bleft
        elif h1_area==1:
            human1_.afront,human1_.bright = h1_changea if h1_changea < human1_.afront else human1_.afront, h1_changeb if h1_changeb < human1_.bright else human1_.bright
        elif h1_area==2:
            human1_.aback,human1_.bleft = h1_changea if h1_changea < human1_.aback else human1_.aback, h1_changeb if h1_changeb < human1_.bleft else human1_.bleft
        elif h1_area==3:
            human1_.aback, human1_.bright = h1_changea if h1_changea < human1_.aback else human1_.aback, h1_changeb if h1_changeb < human1_.bright else human1_.bright


    return squeeze


