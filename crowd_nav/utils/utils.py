class AverageMeter():

    def __init__(self):

        self.reset()

    def reset(self):

        self.count=0
        self.newval=0
        self.sum=0
        self.avg=0

    def update(self,newval,n=1):

        self.newval=newval
        self.sum+=newval*n
        self.count+=n
        self.avg=self.sum/self.count