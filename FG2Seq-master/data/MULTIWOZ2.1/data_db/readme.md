# db 数据

1.   all_db.json

     -   我做的全量数据集；
     -   key 字段中，domain 不能作为 entity 属性，要放外面作为对话开头领域；其它 value 为 dontknow 的有一部分是这个领域有这个属性，但这个实体没有，所以不知道，还有一部分是 paddIng 的其它领域的属性，所以不知道，个人建议可以把 dontknow 都删了；

2.   其它 json

     -   官方的数据库文件；
     -   key 字段很多没在 dfnet 中用到，得肉眼去查要用哪些，不建议使用；

     # 怎么做

     1.   个人建议把数据集的 txt 文件中，那 7 个实体换成格式相同的 222 个全量实体；
     2.   注意把实体中空格改为下划线；
     3.   dfnet 数据库中的实体会有 ref 和 choice 这两个属性，这个在每个对话的实体中可能有也可能没有而且各对话不同，所以如果某个对话中的实体有这两个属性，请把该对话对应的 222 个实体都加上这两个属性和其属性值；（开会忘记说了）

     # 目的

     这个任务的目的就是尽可能相同的把 dfnet 中的 7 个实体换成 222 个全量实体，不能缺属性，也不要多属性

     